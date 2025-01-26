from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from knn.graph import Graph


class FVHD:
    def __init__(
        self,
        n_components: int = 2,
        nn: int = 2,
        rn: int = 1,
        c: float = 0.1,
        optimizer: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Dict[str, Any] = None,
        epochs: int = 200,
        eta: float = 0.1,
        device: str = "cpu",
        graph_file: str = "",
        autoadapt=False,
        velocity_limit=False,
        verbose=True,
        mutual_neighbors_epochs: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.eta = eta
        self.a = 0.9
        self.b = 0.3
        self.device = device
        self.verbose = verbose
        self.graph_file = graph_file
        self._current_epoch = 0

        self.autoadapt = autoadapt
        self.buffer_len = 10
        self.curr_max_velo = torch.tensor(([0.0] * self.buffer_len))
        self.curr_max_velo_idx = 1
        self.velocity_limit = velocity_limit
        self.max_velocity = 1.0
        self.vel_dump = 0.95
        self.x = None
        self.delta_x = None
        self.mutual_neighbors_epochs = mutual_neighbors_epochs

    def fit_transform(self, X: torch.Tensor, graphs: list[Graph]) -> np.ndarray:
        x = X.to(self.device)
        graph = graphs[0]
        nn = torch.tensor(graph.indexes[:, : self.nn].astype(np.int32))
        nn = nn.to(self.device)
        n = x.shape[0]
        rn = torch.randint(0, n, (n, self.rn)).to(self.device)
        nn = nn.reshape(-1)
        rn = rn.reshape(-1)

        if self.optimizer is None:
            return self.force_directed_method(x, nn, rn, graphs)
        return self.optimizer_method(x.shape[0], nn, rn)

    def optimizer_method(self, N, NN, RN):
        if self.x is None:
            self.x = torch.rand(
                (N, 1, self.n_components), requires_grad=True, device=self.device
            )
        optimizer = self.optimizer(params={self.x}, **self.optimizer_kwargs)
        for i in range(self.epochs):
            loss = self.optimizer_step(optimizer, NN, RN)
            if loss < 1e-10:
                return self.x[:, 0].detach()
            if self.verbose:
                print(f"\r{i} loss: {loss.item()}, X: {self.x[0]}", end="")
                if i % 100 == 0:
                    print()

        return self.x[:, 0].detach().cpu().numpy()

    def _calculate_distances(self, indices):
        diffs = self.x - torch.index_select(self.x, 0, indices).view(
            self.x.shape[0], -1, self.n_components
        )
        dist = torch.sqrt(
            torch.sum((diffs + 1e-8) * (diffs + 1e-8), dim=-1, keepdim=True)
        )
        return diffs, dist

    def optimizer_step(self, optimizer, NN, RN) -> Tensor:
        optimizer.zero_grad()
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean(
            (1 - rn_dist) * (1 - rn_dist)
        )
        loss.backward()
        optimizer.step()
        return loss

    def force_directed_method(
        self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor, graphs: list[Graph]
    ) -> np.ndarray:
        nn_new = NN.reshape(X.shape[0], self.nn, 1)
        nn_new = [nn_new for _ in range(self.n_components)]
        nn_new = torch.cat(nn_new, dim=-1).to(torch.long)

        rn_new = RN.reshape(X.shape[0], self.rn, 1)
        rn_new = [rn_new for _ in range(self.n_components)]
        rn_new = torch.cat(rn_new, dim=-1).to(torch.long)

        if self.x is None:
            self.x = torch.rand((X.shape[0], 1, self.n_components), device=self.device)
        if self.delta_x is None:
            self.delta_x = torch.zeros_like(self.x)

        for i in range(self.epochs):
            self._current_epoch = i
            loss = self.__force_directed_step(NN, RN, nn_new, rn_new, graphs)
            if self.verbose and i % 100 == 0:
                print(f"\r{i} loss: {loss.item()}")

        return self.x[:, 0].cpu().numpy()

    def __force_directed_step(self, NN, RN, NN_new, RN_new, graphs):
        if self.mutual_neighbors_epochs and self.epochs - self._current_epoch <= self.mutual_neighbors_epochs:
            graph = graphs[1]
            NN = torch.tensor(graph.indexes[:, : self.nn].astype(np.int32)).to(self.device).reshape(-1)
            NN_new = NN.reshape(self.x.shape[0], self.nn, 1)
            NN_new = torch.cat([NN_new for _ in range(self.n_components)], dim=-1).to(torch.long)

        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        f_nn, f_rn = self.__compute_forces(rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new)

        f = -f_nn - self.c * f_rn
        self.delta_x = self.a * self.delta_x + self.b * f
        squared_velocity = torch.sum(self.delta_x * self.delta_x, dim=-1)
        sqrt_velocity = torch.sqrt(squared_velocity)

        if self.velocity_limit:
            self.delta_x[squared_velocity > self.max_velocity ** 2] *= (
                    self.max_velocity
                    / sqrt_velocity[squared_velocity > self.max_velocity ** 2]
            ).reshape(-1, 1)

        self.x += self.eta * self.delta_x

        if self.autoadapt:
            self._auto_adaptation(sqrt_velocity)

        if self.velocity_limit:
            self.delta_x *= self.vel_dump

        loss = torch.mean(nn_dist ** 2) + self.c * torch.mean((1 - rn_dist) ** 2)
        return loss

    def _auto_adaptation(self, sqrt_velocity):
        v_avg = self.delta_x.mean()
        self.curr_max_velo[self.curr_max_velo_idx] = sqrt_velocity.max()
        self.curr_max_velo_idx = (self.curr_max_velo_idx + 1) % self.buffer_len
        v_max = self.curr_max_velo.mean()
        if v_max > 10 * v_avg:
            self.eta /= 1.01
        elif v_max < 10 * v_avg:
            self.eta *= 1.01
        if self.eta < 0.01:
            self.eta = 0.01

    def __compute_forces(self, rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new):
        if self.mutual_neighbors_epochs and self.epochs - self._current_epoch <= self.mutual_neighbors_epochs:
            nn_attraction = 1.0 / (nn_dist + 1e-8)
            f_nn = nn_attraction * nn_diffs
        else:
            f_nn = nn_diffs
        f_rn = (rn_dist - 1) / (rn_dist + 1e-8) * rn_diffs

        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(src=f_nn, dim=0, index=NN_new)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(src=f_rn, dim=0, index=RN_new)

        f_nn -= minus_f_nn
        f_rn -= minus_f_rn
        f_nn = torch.sum(f_nn, dim=1, keepdim=True)
        f_rn = torch.sum(f_rn, dim=1, keepdim=True)
        return f_nn, f_rn
