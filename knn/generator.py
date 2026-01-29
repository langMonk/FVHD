from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from .base import GraphData
from .graph import Graph


@dataclass
class NeighborConfig:
    """Configuration for neighbor generation."""

    metric: str = "euclidean"
    n_jobs: int = -1


class NeighborGenerator:
    """
    Generates K-Nearest Neighbors graphs and mutual neighbor graphs.
    """

    def __init__(self, df: pd.DataFrame, config: NeighborConfig):
        """
        Initialize the generator.

        Args:
            df (pd.DataFrame): Input data.
            config (NeighborConfig): Configuration for nearest neighbors search.
        """
        self.config = config
        self._process_input_data(df)
        self.indexes: NDArray | None = None
        self.distances: NDArray | None = None
        self.nn: int | None = None

    def _process_input_data(self, df: pd.DataFrame) -> None:
        self.X = df.to_numpy(dtype="float32")
        self.N = len(df)

    def run(self, nn: int = 100) -> tuple[Graph, Graph]:
        """
        Computes the nearest neighbor and mutual neighbor graphs.

        Args:
            nn (int): Number of nearest neighbors.

        Returns:
            Tuple[Graph, Graph]: The standard KNN graph and the mutual KNN graph.
        """
        self.nn = nn
        nbrs = NearestNeighbors(
            n_neighbors=nn + 1, metric=self.config.metric, n_jobs=self.config.n_jobs
        ).fit(self.X)
        self.distances, self.indexes = nbrs.kneighbors(self.X)

        adj_matrix = np.zeros((self.N, self.N), dtype=bool)
        np.put_along_axis(adj_matrix, self.indexes, True, axis=1)

        mutual_mask = adj_matrix & adj_matrix.T

        mutual_indexes = np.zeros((self.N, nn + 1), dtype=np.int64)
        mutual_distances = np.zeros((self.N, nn + 1), dtype=np.float32)

        target_count = nn + 1

        for i in range(self.N):
            row_mutual_indices = np.where(mutual_mask[i])[0]

            curr_len = len(row_mutual_indices)
            if curr_len < target_count:
                if curr_len == 0:
                    padded_indices = np.array([i] * target_count, dtype=np.int64)
                else:
                    padded_indices = np.pad(
                        row_mutual_indices, (0, target_count - curr_len), mode="edge"
                    )
                mutual_indexes[i] = padded_indices
            else:
                mutual_indexes[i] = row_mutual_indices[:target_count]

            neighbor_is_mutual = mutual_mask[i, self.indexes[i]]
            dists = self.distances[i][neighbor_is_mutual]

            d_len = len(dists)
            if d_len < target_count:
                if d_len == 0:
                    padded_dists = np.zeros(target_count, dtype=np.float32)
                else:
                    padded_dists = np.pad(dists, (0, target_count - d_len), mode="edge")
                mutual_distances[i] = padded_dists
            else:
                mutual_distances[i] = dists[:target_count]

        return Graph(GraphData(indexes=self.indexes, distances=self.distances)), Graph(
            GraphData(indexes=mutual_indexes, distances=mutual_distances)
        )

    def save_binary(self, path: Path) -> None:
        """
        Saves the generated graph to a binary file.

        Args:
            path (Path): Destination file path.

        Raises:
            RuntimeError: If the graph has not been generated yet.
        """
        if self.indexes is None or self.distances is None:
            raise RuntimeError("Run search before saving results")

        with open(path, "wb") as f:
            header = f"{self.N};{self.nn};8\n".encode("ascii")
            f.write(header)
            f.write((0x01020304).to_bytes(8, byteorder="little"))

            mask = np.arange(len(self.indexes))[:, None] != self.indexes
            valid_idx_rows, valid_idx_cols = np.where(mask)

            valid_indexes = self.indexes[valid_idx_rows, valid_idx_cols]
            valid_distances = self.distances[valid_idx_rows, valid_idx_cols]

            dt = np.dtype([("idx", "<i8"), ("dist", "<f4")])
            combined = np.empty(len(valid_indexes), dtype=dt)
            combined["idx"] = valid_indexes
            combined["dist"] = valid_distances.astype(np.float32)

            f.write(combined.tobytes())
