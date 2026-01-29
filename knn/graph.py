from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import GraphData


class Graph:
    """
    Represents a K-Nearest Neighbors graph.

    Attributes:
        indexes (Optional[NDArray]): Indices of neighbors for each data point.
        distances (Optional[NDArray]): Distances to neighbors for each data point.
    """

    def __init__(self, data: GraphData | None = None):
        """
        Initializes the Graph with optional data.

        Args:
            data (Optional[GraphData]): Initial graph data containing
            indexes and distances.
        """
        self.indexes: NDArray | None = data.indexes if data else None
        self.distances: NDArray | None = data.distances if data else None

    def get_neighbors(self, n: int) -> NDArray:
        """
        Retrieves the neighbors for the n-th data point.

        Args:
            n (int): Index of the data point.

        Returns:
            NDArray: Array of neighbor indices.

        Raises:
            ValueError: If the graph is not initialized.
        """
        if self.indexes is None:
            raise ValueError("Graph not initialized")
        return self.indexes[n]

    def load_binary(self, path: Path, nn_count: int) -> None:
        """
        Loads graph data from a binary file.

        Args:
            path (Path): Path to the binary file.
            nn_count (int): Number of nearest neighbors to load per point.
        """
        with open(path, "rb") as f:
            self._read_header(f)
            self._load_binary_data(f, nn_count)

    @staticmethod
    def _read_header(file) -> tuple[int, int]:
        """Reads and validates the file header."""
        header = file.readline().decode("ascii").split(";")
        data_count, overall_nn_count, _ = map(int, header)

        magic_number = int.from_bytes(file.read(8), byteorder="little")
        if magic_number != 0x01020304:
            raise ValueError("Invalid file format")

        return data_count, overall_nn_count

    def _load_binary_data(self, file, nn_count: int) -> None:
        """Loads the binary data body."""
        data_count, overall_nn_count = self._read_header(file)

        self.indexes = np.empty([data_count, nn_count], dtype=np.int64)
        self.distances = np.empty([data_count, nn_count], dtype=np.float32)

        total_entries = data_count * overall_nn_count
        current_row = current_col = 0

        for _ in range(total_entries):
            idx = int.from_bytes(file.read(8), byteorder="little")
            distance = np.frombuffer(file.read(4), dtype=np.float32)[0]

            if current_col < nn_count:
                self.indexes[current_row, current_col] = idx
                self.distances[current_row, current_col] = distance

            current_col += 1
            if current_col == overall_nn_count:
                current_row += 1
                current_col = 0

    def get_conflicting_neighbors(self, labels: pd.Series) -> pd.Series:
        """
        Identifies data points whose neighbors have different labels.

        Args:
            labels (pd.Series): Labels for each data point.

        Returns:
            pd.Series: Indices of data points with conflicting neighbors.

        Raises:
            ValueError: If the graph is not initialized.
        """
        if self.indexes is None:
            raise ValueError("Graph not initialized")

        label_array = labels.to_numpy()
        source_labels = label_array[np.arange(len(self.indexes))]
        neighbor_labels = label_array[self.indexes.astype(int)]

        conflicts = np.any(source_labels[:, np.newaxis] != neighbor_labels, axis=1)
        return pd.Series(np.where(conflicts)[0])
