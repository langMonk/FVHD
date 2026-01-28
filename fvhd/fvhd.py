from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim import Optimizer

from .config import FVHDConfig


class FVHD(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 2,
        nn: int = 5,
        rn: int = 2,
        c: float = 0.2,
        optimizer: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Dict[str, Any] = None,
        epochs: int = 2000,
        eta: float = 0.2,
        device: str = "cpu",
        autoadapt: bool = True,
        velocity_limit: bool = True,
        verbose: bool = True,
        mutual_neighbors_epochs: Optional[int] = 300,
        metric: str = "euclidean",
        n_jobs: int = -1,
        config: Optional[FVHDConfig] = None
    ) -> None:
        if config:
            self.config = config
        else:
            self.config = FVHDConfig(
                n_components=n_components,
                nn=nn,
                rn=rn,
                c=c,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                epochs=epochs,
                eta=eta,
                device=device,
                autoadapt=autoadapt,
                velocity_limit=velocity_limit,
                verbose=verbose,
                mutual_neighbors_epochs=mutual_neighbors_epochs,
                metric=metric,
                n_jobs=n_jobs,
            )
        
        # Expose parameters for sklearn compliance
        self.n_components = self.config.n_components
        self.nn = self.config.nn
        self.rn = self.config.rn
        self.c = self.config.c
        self.optimizer = self.config.optimizer
        self.optimizer_kwargs = self.config.optimizer_kwargs
        self.epochs = self.config.epochs
        self.eta = self.config.eta
        self.device = self.config.device
        self.autoadapt = self.config.autoadapt
        self.velocity_limit = self.config.velocity_limit
        self.verbose = self.config.verbose
        self.mutual_neighbors_epochs = self.config.mutual_neighbors_epochs
        self.metric = self.config.metric
        self.n_jobs = self.config.n_jobs

        self.embedding_ = None
        self._x = None
        self._delta_x = None
        self._current_epoch = 0
        self._buffer_len = 10
        self._curr_max_velo = None
        self._curr_max_velo_idx = 0
        self._max_velocity = 1.0
        self._vel_dump = 0.95
        self._a = 0.9
        self._b = 0.3

    def fit(self, X: Union[np.ndarray, torch.Tensor], y=None, **kwargs):
        self.fit_transform(X, y, **kwargs)
        return self

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], y=None, 
                      nn_idx: Optional[np.ndarray] = None, 
                      rn_idx: Optional[np.ndarray] = None,
                      mutual_idx: Optional[np.ndarray] = None) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # 1. Generate or Use Provided Graphs
        if nn_idx is not None:
             # Use provided indices
             # We expect nn_idx, mutual_idx if mutual epochs > 0
             # Generate distances if needed? 
             # The optimizer needs distances. _calculate_distances computes them from X and neighbor indices.
             # So we just need indices!
             pass
        else:
             nn_idx, _, mutual_idx, _ = self._compute_graphs(X_np)

        x_data = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        self._n_samples = x_data.shape[0]

        nn_tensor = torch.tensor(nn_idx[:, :self.nn].astype(np.int32)).to(self.device)
        
        if rn_idx is not None:
             rn_tensor = torch.tensor(rn_idx[:, :self.rn].astype(np.int32)).to(self.device)
        else:
             rn_tensor = torch.randint(0, self._n_samples, (self._n_samples, self.rn)).to(self.device)
        
        nn_tensor_flat = nn_tensor.reshape(-1)
        rn_tensor_flat = rn_tensor.reshape(-1)

        if self.optimizer is None:
            self.embedding_ = self._force_directed_method(
                x_data, nn_tensor_flat, rn_tensor_flat, 
                mutual_idx
            )
        else:
            self.embedding_ = self._optimizer_method(self._n_samples, nn_tensor_flat, rn_tensor_flat)
            
        return self.embedding_

    def _compute_graphs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.nn + 1, metric=self.metric, n_jobs=self.n_jobs).fit(X)
        distances, indexes = nbrs.kneighbors(X)

        adj_matrix = np.zeros((n_samples, n_samples), dtype=bool)
        np.put_along_axis(adj_matrix, indexes, True, axis=1)
        
        mutual_mask = adj_matrix & adj_matrix.T

        mutual_indexes = np.zeros((n_samples, self.nn + 1), dtype=np.int64)
        mutual_distances = np.zeros((n_samples, self.nn + 1), dtype=np.float32)

        for i in range(n_samples):
            mutual_idx = np.where(mutual_mask[i])[0]
            current_count = len(mutual_idx)
            target_count = self.nn + 1
            
            if current_count < target_count:
                if current_count == 0:
                     mutual_idx = np.array([i] * target_count)
                else:
                     mutual_idx = np.pad(mutual_idx, (0, target_count - current_count), mode='edge')
            
            mutual_indexes[i] = mutual_idx[:target_count]
            
            dist_mask = np.isin(indexes[i], mutual_idx[:target_count])
            dists = distances[i][dist_mask]
            
            if len(dists) < target_count:
                 if len(dists) == 0:
                      dists = np.zeros(target_count)
                 else:
                      dists = np.pad(dists, (0, target_count - len(dists)), mode='edge')
            
            mutual_distances[i] = dists[:target_count]

        return indexes, distances, mutual_indexes, mutual_distances

    def _optimizer_method(self, N, NN, RN):
        if self._x is None:
            self._x = torch.rand(
                (N, 1, self.n_components), requires_grad=True, device=self.device
            )
        
        if isinstance(self.optimizer, type):
             optimizer_instance = self.optimizer(params=[self._x], **(self.optimizer_kwargs or {}))
        else:
             raise ValueError("Optimizer should be a class type")

        for i in range(self.epochs):
            loss = self._optimizer_step(optimizer_instance, NN, RN)
            if loss < 1e-10:
                return self._x[:, 0].detach().cpu().numpy()
            if self.verbose:
                print(f"\r{i} loss: {loss.item()}", end="")
        
        if self.verbose:
            print()

        return self._x[:, 0].detach().cpu().numpy()

    def _optimizer_step(self, optimizer, NN, RN) -> Tensor:
        optimizer.zero_grad()
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean(
            (1 - rn_dist) * (1 - rn_dist)
        )
        loss.backward()
        optimizer.step()
        return loss

    def _calculate_distances(self, indices):
        target_points = torch.index_select(self._x, 0, indices).view(
            self._x.shape[0], -1, self.n_components
        )
        diffs = self._x - target_points
        dist = torch.sqrt(
            torch.sum((diffs + 1e-8) * (diffs + 1e-8), dim=-1, keepdim=True)
        )
        return diffs, dist

    def _force_directed_method(
        self, X_tensor: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor, mutual_indexes: np.ndarray
    ) -> np.ndarray:
        nn_new = NN.reshape(X_tensor.shape[0], self.nn, 1)
        nn_new = nn_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components).to(torch.long)

        rn_new = RN.reshape(X_tensor.shape[0], self.rn, 1)
        rn_new = rn_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components).to(torch.long)

        if self._x is None:
            self._x = torch.rand((X_tensor.shape[0], 1, self.n_components), device=self.device)
        if self._delta_x is None:
            self._delta_x = torch.zeros_like(self._x)
            
        self._curr_max_velo = torch.zeros(self._buffer_len, device=self.device)

        for i in range(self.epochs):
            self._current_epoch = i
            
            current_NN = NN
            current_NN_new = nn_new
            
            if self.mutual_neighbors_epochs and (self.epochs - i <= self.mutual_neighbors_epochs) and mutual_indexes is not None:
                 mutual_nn = torch.tensor(mutual_indexes[:, :self.nn].astype(np.int32)).to(self.device).reshape(-1)
                 current_NN = mutual_nn
                 current_NN_new = current_NN.reshape(X_tensor.shape[0], self.nn, 1)
                 current_NN_new = current_NN_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components)
                 current_NN_new = current_NN_new.to(torch.long)

            loss = self.__force_directed_step(current_NN, RN, current_NN_new, rn_new)
            
            if self.verbose and i % 100 == 0:
                print(f"\rEpoch {i}/{self.epochs} loss: {loss.item():.4f}", end="")

        if self.verbose:
            print()
            
        return self._x[:, 0].cpu().numpy()

    def __force_directed_step(self, NN, RN, NN_new, RN_new):
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        f_nn, f_rn = self.__compute_forces(rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new)

        f = -f_nn - self.c * f_rn
        self._delta_x = self._a * self._delta_x + self._b * f
        
        squared_velocity = torch.sum(self._delta_x * self._delta_x, dim=-1)
        sqrt_velocity = torch.sqrt(squared_velocity)

        if self.velocity_limit:
            mask = squared_velocity > self._max_velocity ** 2
            if mask.any():
                scale = self._max_velocity / (sqrt_velocity[mask] + 1e-8)
                self._delta_x[mask] *= scale.reshape(-1, 1)

        self._x += self.eta * self._delta_x

        if self.autoadapt:
            self._auto_adaptation(sqrt_velocity)

        if self.velocity_limit:
            self._delta_x *= self._vel_dump

        loss = torch.mean(nn_dist ** 2) + self.c * torch.mean((1 - rn_dist) ** 2)
        return loss

    def _auto_adaptation(self, sqrt_velocity):
        v_avg = self._delta_x.mean()
        self._curr_max_velo[self._curr_max_velo_idx] = sqrt_velocity.max()
        self._curr_max_velo_idx = (self._curr_max_velo_idx + 1) % self._buffer_len
        v_max = self._curr_max_velo.mean()
        
        if v_max > 10 * v_avg:
            self.eta /= 1.01
        elif v_max < 10 * v_avg:
            self.eta *= 1.01
            
        if self.eta < 0.01:
            self.eta = 0.01

    def __compute_forces(self, rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new):
        is_mutual_phase = self.mutual_neighbors_epochs and (self.epochs - self._current_epoch <= self.mutual_neighbors_epochs)
        
        if is_mutual_phase:
             nn_attraction = 1.0 / (nn_dist + 1e-8)
             f_nn = nn_attraction * nn_diffs
        else:
             f_nn = nn_diffs

        f_rn = (rn_dist - 1) / (rn_dist + 1e-8) * rn_diffs

        # Correct scatter_add implementation preserving dimensions logic
        # Forces are vectors (C components)
        
        # We need to flatten NN_new to use it as index for scatter_add on dim=0 if we flatten f_nn to (N*K, C)
        # But we want to accumulate into (N, C) ideally? No, original accumulated into (N, NN, C) then summed.
        # But wait, logic: 
        # Node i is source. Neighbor k (index NN[i,k]) is target.
        # Force F_{ik} acts on i. 
        # Reaction -F_{ik} acts on neighbor k.
        # We want to add -F_{ik} to the force accumulator for node k.
        
        # New approach to ensure correctness without complex 3D scatter:
        # Flatten everything to (TotalInteractions, C)
        
        N = self._n_samples
        C = self.n_components
        
        # f_nn is (N, NN, C) effectively (from view logic in calculate_distances)
        # Let's reshape to (N*NN, C)
        f_nn_flat = f_nn.reshape(-1, C)
        f_rn_flat = f_rn.reshape(-1, C)
        
        # NN_new is (N*NN, C) - this was my expand logic.
        # Actually for scatter_add on dim=0, we need indices in [0, N-1].
        # NN_new contains indices of neighbors.
        # But its shape is (N*NN, C), meaning for each component we have the index.
        # This is correct for scatter_add.
        
        # Accumulate reaction forces
        # We need a tensor of shape (N, C) to hold total force on each node?
        # No, original code structure implied:
        # minus_f_nn = zeros_like(f_nn).scatter_add(..., index=NN_new)
        # This means minus_f_nn has shape (N, NN, C).
        # This means it distributes the reaction forces back into a structure that looks like "forces from neighbors".
        # This is only useful if we then sum over neighbors (dim 1).
        
        # BUT: scatter_add(src, index)
        # If I have force F from i->j. I want to add -F to j.
        # j is at index NN_new.
        # So I put -F at row j.
        # But wait, if I put it at row j, does it land in the column corresponding to i?
        # Original code: `zeros_like(f_nn)` implies shape (N, NN, C).
        # If I scatter to row j, which column k do I land in?
        # `scatter_add_` uses the indices to determine location.
        # If `index` has shape (N, NN, C), then for each element (i, k, c) in src,
        # it adds src[i,k,c] to out[index[i,k,c], k, c].
        # So it preserves column `k`.
        # So the reaction force from i to its k-th neighbor (who is node j)
        # is added to the k-th neighbor slot of node j?
        # This means node j receives a force in its k-th slot.
        # This implies we assume i is the k-th neighbor of j?
        # NO. That assumption is generally false.
        # So the original code's logic of `minus_f_nn` seems to rely on `sum(dim=1)` effectively checking "all slots".
        # Yes, since we sum `f_nn` over dim 1 later (`torch.sum(f_nn, dim=1)`),
        # it doesn't matter which slot k the force lands in, as long as it lands on the correct node row.
        # So `minus_f_nn` accumulates forces acting on nodes, stored somewhat arbitrarily in columns.
        
        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(0, NN_new.view(-1, self.nn, self.n_components), f_nn)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(0, RN_new.view(-1, self.rn, self.n_components), f_rn)
        
        # Note: NN_new was reshaped above in `force_directed_method` to (N*nn, C).
        # We need to reshape it back to (N, nn, C) for this call if f_nn is (N, nn, C).
        # In my refactor:
        # f_nn comes from `_calculate_distances` -> `diffs`.
        # `diffs` is `self.x - target`. `self.x` is (N, 1, C). `target` is (N, K, C).
        # So diffs is (N, K, C).
        
        # So I need NN_new to be (N, K, C).
        # In `force_directed_method`, I did:
        # nn_new = nn_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components) -> (N*K, C).
        # So I should reshape it back for this operation or keep it (N, K, C).
        
        # Better: keep N*K, C view for flat operations? 
        # But scatter_add on (N, K, C) is cleaner to read. 
        # I will cast NN_new to (N, K, C).
        
        NN_new_3d = NN_new.view(self._n_samples, -1, self.n_components)
        RN_new_3d = RN_new.view(self._n_samples, -1, self.n_components)
        
        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(0, NN_new_3d, f_nn)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(0, RN_new_3d, f_rn)

        f_nn -= minus_f_nn
        f_rn -= minus_f_rn
        f_nn = torch.sum(f_nn, dim=1, keepdim=True)
        f_rn = torch.sum(f_rn, dim=1, keepdim=True)
        return f_nn, f_rn
