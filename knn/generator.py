from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from .base import GraphData
from .graph import Graph


@dataclass
class NeighborConfig:
    metric: str = "euclidean"
    n_jobs: int = -1


class NeighborGenerator:
    def __init__(self, df: pd.DataFrame, config: NeighborConfig):
        self.config = config
        self._process_input_data(df)
        self.indexes: Optional[NDArray] = None
        self.distances: Optional[NDArray] = None
        self.nn: Optional[int] = None

    def _process_input_data(self, df: pd.DataFrame) -> None:
        self.X = df.to_numpy(dtype="float32")
        self.N = len(df)

    def run(self, nn: int = 100) -> tuple[Graph, Graph]:
        self.nn = nn
        nbrs = NearestNeighbors(n_neighbors=nn + 1, metric=self.config.metric, n_jobs=self.config.n_jobs).fit(self.X)
        self.distances, self.indexes = nbrs.kneighbors(self.X)

        adj_matrix = np.zeros((self.N, self.N), dtype=bool)
        np.put_along_axis(adj_matrix, self.indexes, True, axis=1)
        mutual_mask = adj_matrix & adj_matrix.T

        mutual_indexes = np.zeros((self.N, nn + 1), dtype=np.int64)
        mutual_distances = np.zeros((self.N, nn + 1), dtype=np.float32)

        for i in range(self.N):
            mutual_idx = np.where(mutual_mask[i])[0]
            if len(mutual_idx) < nn + 1:
                mutual_idx = np.pad(mutual_idx, (0, nn + 1 - len(mutual_idx)), mode='edge')
            mutual_indexes[i] = mutual_idx[:nn + 1]

            dist_mask = np.isin(self.indexes[i], mutual_idx[:nn + 1])
            mutual_distances[i] = np.pad(self.distances[i][dist_mask],
                                         (0, nn + 1 - np.sum(dist_mask)),
                                         mode='edge')

        return Graph(GraphData(indexes=self.indexes, distances=self.distances)), \
            Graph(GraphData(indexes=mutual_indexes, distances=mutual_distances))

    def save_binary(self, path: Path) -> None:
        if self.indexes is None or self.distances is None:
            raise RuntimeError("Run search before saving results")

        with open(path, "wb") as f:
            header = f"{self.N};{self.nn};8\n".encode("ascii")
            f.write(header)
            f.write((0x01020304).to_bytes(8, byteorder="little"))

            mask = np.arange(len(self.indexes))[:, None] != self.indexes
            valid_indices = np.where(mask)

            for i, j in zip(*valid_indices):
                f.write(int(self.indexes[i, j]).to_bytes(8, byteorder="little"))
                f.write(self.distances[i, j].astype(np.float32).tobytes())
