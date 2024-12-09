import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from torch.optim import (
    ASGD,
    LBFGS,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SparseAdam,
)

_optimizer_mapping = {
    cls.__name__.lower(): cls
    for cls in [
        Adadelta,
        Adagrad,
        Adam,
        AdamW,
        SparseAdam,
        Adamax,
        ASGD,
        LBFGS,
        NAdam,
        RAdam,
        RMSprop,
        Rprop,
        SGD,
    ]
}


class IVHDGrad(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 2,
        nn: int = 2,
        rn: int = 1,
        pos_weight: float = 0.5,
        optimizer: str = "adam",
        optimizer_params=None,
        steps: int = 200,
        epsilon: float = 1e-15,
        re_draw_remote_neighbors: bool = False,
        verbose: bool = False,
        distance: str = "euclidean",
    ) -> None:
        if optimizer_params is None:
            optimizer_params = {"lr": 0.01}
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.pos_weight = pos_weight
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.simulation_steps = steps
        self.epsilon = epsilon
        self.re_draw_remote_neighbors = re_draw_remote_neighbors
        self.verbose = verbose
        self.distance = distance

    def transform(self, X, precomputed_nn_indices=None):
        if precomputed_nn_indices is None:
            nns = self._get_nearest_neighbors_indexes(X)
        else:
            self._validate_precomputed_nn_indices(X, precomputed_nn_indices)
            nns = precomputed_nn_indices
        rns = self._get_remote_neighbors_indexes(X)

        x = torch.rand(X.shape[0], self.n_components, requires_grad=True)
        optimizer = _optimizer_mapping[self.optimizer]([x], **self.optimizer_params)

        for step in range(self.simulation_steps):
            neighborhoods = x[nns]
            if self.re_draw_remote_neighbors:
                rns = self._get_remote_neighbors_indexes(X)
            remote_neighborhoods = x[rns]

            xu = x.unsqueeze(1)
            dist_emb_neighbor = self._calculate_distance(
                self.distance, xu, neighborhoods
            )
            dist_emb_remote = self._calculate_distance(
                self.distance, xu, remote_neighborhoods
            )

            del_n = 0  # desired distance between point and its nearest neighbors
            del_r = 1  # desired distance between point and its remote neighbors

            cost_emb_neighbor = ((dist_emb_neighbor - del_n) ** 2).sum()
            cost_emb_remote = ((dist_emb_remote - del_r) ** 2).sum()

            cost = 2 * (
                cost_emb_neighbor * self.pos_weight
                + cost_emb_remote * (1 - self.pos_weight)
            )
            if self.verbose:
                print(f"Step: {step}\tCost: {cost}")

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        return x.detach().numpy()

    def _validate_precomputed_nn_indices(self, X, precomputed_nn_indices):
        if precomputed_nn_indices.shape != (X.shape[0], self.nn):
            raise ValueError(
                f"passed precomputed_nn_indices shape ({precomputed_nn_indices.shape}) does"
                " not match required shape ({(X.shape[0], self.nn)})"
            )

    def _get_nearest_neighbors_indexes(self, X: np.ndarray) -> np.ndarray:
        # for every point in X find indexes of its 'nn' nearest neighbors
        knn_model = NearestNeighbors(n_neighbors=self.nn + 1)
        knn_model.fit(X)
        _, indices = knn_model.kneighbors(X)
        return indices[:, 1:]

    def _get_remote_neighbors_indexes(self, X: np.ndarray) -> np.ndarray:
        # for every point in X sample indices of its 'rn' remote neighbors
        return np.random.randint(low=0, high=X.shape[0], size=(X.shape[0], self.rn))

    @staticmethod
    def __sklearn_is_fitted__() -> bool:
        return True  # all calculations are performed in 'transform' method

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, precomputed_nn_indices=None, **fit_params):
        return self.transform(X, precomputed_nn_indices)

    def _calculate_distance(self, distance_string, x_i, x_k):
        if distance_string == "cosine":
            top = (x_i * x_k).sum(dim=-1)
            bottom_i = torch.norm(x_i, p=2, dim=-1)
            bottom_k = torch.norm(x_k, p=2, dim=-1)
            return 1 - (top / (bottom_i * bottom_k + self.epsilon))

        elif distance_string == "binary":
            return ((x_i - x_k) ** 2).sum(dim=-1)

        else:
            return torch.norm(x_i - x_k, p=2, dim=-1)
