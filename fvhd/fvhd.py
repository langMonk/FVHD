from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim import Optimizer

from .config import FVHDConfig


class FVHD(BaseEstimator, TransformerMixin):
    """
    Fast Visualization of High-Dimensional Data (FVHD).

    Implements a force-directed graph layout algorithm for dimensionality reduction.
    """

    def __init__(
        self,
        n_components: int = 2,
        nn: int = 5,
        rn: int = 2,
        c: float = 0.2,
        optimizer: type[Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        epochs: int = 2000,
        eta: float = 0.2,
        device: str = "cpu",
        autoadapt: bool = True,
        velocity_limit: bool = True,
        verbose: bool = True,
        mutual_neighbors_epochs: int | None = 300,
        metric: str = "euclidean",
        n_jobs: int = -1,
        config: FVHDConfig | None = None,
    ) -> None:
        """
        Initialize FVHD.

        Args:
            n_components (int): Target dimensionality.
            nn (int): Number of nearest neighbors.
            rn (int): Number of random neighbors.
            c (float): Attraction/Repulsion balance coefficient.
            optimizer (Optional[Type[Optimizer]]): PyTorch optimizer class.
            optimizer_kwargs (Dict[str, Any]): Arguments for the optimizer.
            epochs (int): Number of optimization epochs.
            eta (float): Learning rate.
            device (str): Computation device ('cpu', 'cuda', 'mps').
            autoadapt (bool): Enable automatic learning rate adaptation.
            velocity_limit (bool): Enable velocity limiting.
            verbose (bool): Print progress messages.
            mutual_neighbors_epochs (Optional[int])
            metric (str): Distance metric for KNN.
            n_jobs (int): Parallel jobs for KNN search.
            config (Optional[FVHDConfig]): Configuration object.
        """
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

    def fit(self, X: np.ndarray | torch.Tensor, y=None, **kwargs):
        """
        Fit the model to X.

        Args:
            X: Input data.
            y: Ignored.
            **kwargs: Additional arguments passed to fit_transform.

        Returns:
            self
        """
        self.fit_transform(X, y, **kwargs)
        return self

    def fit_transform(self, X: np.ndarray | torch.Tensor, y=None, **fit_params) -> np.ndarray:
        """
        Fit the model to X and return the transformed data.

        Args:
            X: Input data.
            y: Ignored.
            **fit_params: Additional parameters including:
                - nn_idx (Optional[np.ndarray]): Precomputed nearest neighbor idxs.
                - rn_idx (Optional[np.ndarray]): Precomputed random neighbor idxs.
                - mutual_idx (Optional[np.ndarray]): Precomputed mn idxs.
                - graph (Optional[Graph]): Graph object containing nearest idxs.

        Returns:
            np.ndarray: The embedding.
        """
        nn_idx: np.ndarray | None = fit_params.get("nn_idx")
        rn_idx: np.ndarray | None = fit_params.get("rn_idx")
        mutual_idx: np.ndarray | None = fit_params.get("mutual_idx")
        graph = fit_params.get("graph")  # type: Graph | None
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        if graph is not None:
            if hasattr(graph, "indexes") and graph.indexes is not None:
                nn_idx = graph.indexes

        if nn_idx is not None:
            if mutual_idx is None and self.mutual_neighbors_epochs:
                _, _, mutual_idx, _ = self._compute_graphs(X_np, precomputed_nn=nn_idx)
        else:
            nn_idx, _, mutual_idx, _ = self._compute_graphs(X_np)

        x_data = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        self._n_samples = x_data.shape[0]

        nn_tensor = torch.tensor(nn_idx[:, : self.nn].astype(np.int32)).to(self.device)

        if rn_idx is not None:
            rn_tensor = torch.tensor(rn_idx[:, : self.rn].astype(np.int32)).to(self.device)
        else:
            rn_tensor = torch.randint(0, self._n_samples, (self._n_samples, self.rn)).to(
                self.device
            )

        nn_tensor_flat = nn_tensor.reshape(-1)
        rn_tensor_flat = rn_tensor.reshape(-1)

        if self.optimizer is None:
            # Ensure mutual_idx is not None for type checker
            mutual_idx_safe = mutual_idx if mutual_idx is not None else np.array([], dtype=np.int64)
            self.embedding_ = self._force_directed_method(
                x_data, nn_tensor_flat, rn_tensor_flat, mutual_idx_safe
            )
        else:
            self.embedding_ = self._optimizer_method(
                self._n_samples, nn_tensor_flat, rn_tensor_flat
            )

        return self.embedding_

    def _compute_graphs(
        self, X: np.ndarray, precomputed_nn: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes NN and Mutual NN graphs."""
        n_samples = X.shape[0]

        if precomputed_nn is None:
            nbrs = NearestNeighbors(
                n_neighbors=self.nn + 1, metric=self.metric, n_jobs=self.n_jobs
            ).fit(X)
            distances, indexes = nbrs.kneighbors(X)
        else:
            indexes = precomputed_nn
            distances = np.zeros_like(indexes, dtype=np.float32)

        adj_matrix = np.zeros((n_samples, n_samples), dtype=bool)
        np.put_along_axis(adj_matrix, indexes, True, axis=1)

        mutual_mask = adj_matrix & adj_matrix.T

        mutual_indexes = np.zeros((n_samples, self.nn + 1), dtype=np.int64)
        mutual_distances = np.zeros((n_samples, self.nn + 1), dtype=np.float32)

        target_count = self.nn + 1

        for i in range(n_samples):
            mutual_idx = np.where(mutual_mask[i])[0]
            curr_len = len(mutual_idx)

            if curr_len < target_count:
                if curr_len == 0:
                    padded_idx = np.array([i] * target_count, dtype=np.int64)
                else:
                    padded_idx = np.pad(mutual_idx, (0, target_count - curr_len), mode="edge")
                mutual_indexes[i] = padded_idx
            else:
                mutual_indexes[i] = mutual_idx[:target_count]

            if precomputed_nn is None:
                neighbor_is_mutual = mutual_mask[i, indexes[i]]
                dists = distances[i][neighbor_is_mutual]

                d_len = len(dists)
                if d_len < target_count:
                    if d_len == 0:
                        padded_dists = np.zeros(target_count, dtype=np.float32)
                    else:
                        padded_dists = np.pad(dists, (0, target_count - d_len), mode="edge")
                    mutual_distances[i] = padded_dists
                else:
                    mutual_distances[i] = dists[:target_count]

        return indexes, distances, mutual_indexes, mutual_distances

    def _optimizer_method(self, N, NN, RN):
        """Optimization using PyTorch optimizers."""
        if self._x is None:
            self._x = torch.rand((N, 1, self.n_components), requires_grad=True, device=self.device)

        if isinstance(self.optimizer, type):
            optimizer_instance = self.optimizer(params=[self._x], **(self.optimizer_kwargs or {}))
        else:
            raise ValueError("Optimizer should be a class type")

        for i in range(self.epochs):
            loss = self._optimizer_step(optimizer_instance, NN, RN)
            if loss < 1e-10:
                return self._x[:, 0].detach().cpu().numpy()
            if self.verbose and i % 100 == 0:
                print(f"\r{i} loss: {loss.item():.4f}", end="")

        if self.verbose:
            print()

        return self._x[:, 0].detach().cpu().numpy()

    def _optimizer_step(self, optimizer, NN, RN) -> Tensor:
        optimizer.zero_grad()
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean((1 - rn_dist) * (1 - rn_dist))
        loss.backward()
        optimizer.step()
        return loss

    def _calculate_distances(self, indices: torch.Tensor):
        assert self._x is not None, "self._x must be initialized"
        target_points = torch.index_select(self._x, 0, indices.long()).view(
            self._x.shape[0], -1, self.n_components
        )
        diffs = self._x - target_points
        dist = torch.sqrt(torch.sum((diffs + 1e-8) * (diffs + 1e-8), dim=-1, keepdim=True))
        return diffs, dist

    def _force_directed_method(
        self,
        X_tensor: torch.Tensor,
        NN: torch.Tensor,
        RN: torch.Tensor,
        mutual_indexes: np.ndarray,
    ) -> np.ndarray:
        """
        Force-directed graph layout optimization.

        This is the core custom implementation of the FVHD algorithm. It does not use
        autograd but manually computes forces and updates positions using a
        momentum-based update rule with velocity limiting and auto-adaptive
        learning rate.

        Args:
            X_tensor (torch.Tensor): Input data tensor (N x D).
            NN (torch.Tensor): Nearest neighbor indices (N x k).
            RN (torch.Tensor): Random neighbor indices (N x r).
            mutual_indexes (np.ndarray): Indices of mutual nearest neighbors.

        Returns:
            np.ndarray: The resulting embedding (N x n_components).
        """
        nn_new = NN.reshape(X_tensor.shape[0], self.nn, 1)
        nn_new = (
            nn_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components).to(torch.long)
        )

        rn_new = RN.reshape(X_tensor.shape[0], self.rn, 1)
        rn_new = (
            rn_new.expand(-1, -1, self.n_components).reshape(-1, self.n_components).to(torch.long)
        )

        if self._x is None:
            self._x = torch.rand((X_tensor.shape[0], 1, self.n_components), device=self.device)
        if self._delta_x is None:
            self._delta_x = torch.zeros_like(self._x)

        self._curr_max_velo = torch.zeros(self._buffer_len, device=self.device)

        for i in range(self.epochs):
            self._current_epoch = i

            current_NN = NN
            current_NN_new = nn_new

            if (
                self.mutual_neighbors_epochs
                and (self.epochs - i <= self.mutual_neighbors_epochs)
                and mutual_indexes is not None
            ):
                mutual_nn = (
                    torch.tensor(mutual_indexes[:, : self.nn].astype(np.int32))
                    .to(self.device)
                    .reshape(-1)
                )
                current_NN = mutual_nn
                current_NN_new = current_NN.reshape(X_tensor.shape[0], self.nn, 1)
                current_NN_new = current_NN_new.expand(-1, -1, self.n_components).reshape(
                    -1, self.n_components
                )
                current_NN_new = current_NN_new.to(torch.long)

            loss = self.__force_directed_step(current_NN, RN, current_NN_new, rn_new)

            if self.verbose and i % 100 == 0:
                print(f"\rEpoch {i}/{self.epochs} loss: {loss.item():.4f}", end="")

        if self.verbose:
            print()

        return self._x[:, 0].cpu().numpy()

    def __force_directed_step(self, NN, RN, NN_new, RN_new):
        """
        Perform a single step of force-directed optimization.

        Returns:
             Tensor: The scalar loss value for this step.
        """
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        f_nn, f_rn = self.__compute_forces(rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new)

        f = -f_nn - self.c * f_rn
        assert self._delta_x is not None, "self._delta_x must be initialized"
        self._delta_x = self._a * self._delta_x + self._b * f

        squared_velocity = torch.sum(self._delta_x * self._delta_x, dim=-1)
        sqrt_velocity = torch.sqrt(squared_velocity)

        if self.velocity_limit:
            mask = squared_velocity > self._max_velocity**2
            if mask.any():
                scale = self._max_velocity / (sqrt_velocity[mask] + 1e-8)
                self._delta_x[mask] *= scale.reshape(-1, 1)

        self._x += self.eta * self._delta_x

        if self.autoadapt:
            self._auto_adaptation(sqrt_velocity)

        if self.velocity_limit:
            self._delta_x *= self._vel_dump

        loss = torch.mean(nn_dist**2) + self.c * torch.mean((1 - rn_dist) ** 2)
        return loss

    def _auto_adaptation(self, sqrt_velocity):
        """
        Automatically adapt learning rate (eta) based on system energy (velocity).
        """
        assert self._delta_x is not None, "self._delta_x must be initialized"
        assert self._curr_max_velo is not None, "self._curr_max_velo must be initialized"
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
        """
        Compute total forces acting on each point.
        """
        is_mutual_phase = self.mutual_neighbors_epochs and (
            self.epochs - self._current_epoch <= self.mutual_neighbors_epochs
        )

        if is_mutual_phase:
            nn_attraction = 1.0 / (nn_dist + 1e-8)
            f_nn = nn_attraction * nn_diffs
        else:
            f_nn = nn_diffs

        f_rn = (rn_dist - 1) / (rn_dist + 1e-8) * rn_diffs

        NN_new_3d = NN_new.view(self._n_samples, -1, self.n_components)
        RN_new_3d = RN_new.view(self._n_samples, -1, self.n_components)

        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(0, NN_new_3d, f_nn)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(0, RN_new_3d, f_rn)

        f_nn -= minus_f_nn
        f_rn -= minus_f_rn
        f_nn = torch.sum(f_nn, dim=1, keepdim=True)
        f_rn = torch.sum(f_rn, dim=1, keepdim=True)
        return f_nn, f_rn
