from dataclasses import dataclass
from datetime import datetime
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

    def run(self, nn: int = 100) -> Graph:
        self.nn = nn
        nbrs = NearestNeighbors(
            n_neighbors=nn + 1, metric=self.config.metric, n_jobs=self.config.n_jobs
        ).fit(self.X)

        start = datetime.now()
        self.distances, self.indexes = nbrs.kneighbors(self.X)
        print(f"Search completed in {datetime.now() - start}")

        return Graph(GraphData(indexes=self.indexes, distances=self.distances))

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
