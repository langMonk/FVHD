import struct

import numpy as np
import pandas as pd
from typing import Optional

class Graph:
    def __init__(self):
        self.indexes : Optional[np.ndarray] = None
        self.distances : Optional[np.ndarray] = None

    def get_neighbors(self, n: int) -> np.ndarray:
        return self.indexes[n]

    def load_from_binary_file(self, input_file_path: str, nn_count: int) -> None:
        with open(input_file_path, "rb") as f:
            data_count, overall_nn_count, _ = [
                int(x) for x in f.readline().decode("ascii").split(sep=";")
            ]

            assert 0x01020304 == int.from_bytes(f.read(8), byteorder="little")

            self.indexes = np.empty([data_count, nn_count])
            self.distances = np.empty([data_count, nn_count])

            k = 0
            j = 0
            for i in range(0, data_count*overall_nn_count):
                data = int.from_bytes(f.read(8), byteorder="little")
                distance = struct.unpack("f", f.read(4))[0]
                if i > 0 and i % overall_nn_count == 0:
                    j = 0
                    k += 1
                if j < nn_count:
                    self.indexes[k][j] = int(data)
                    self.distances[k][j] = float(distance)
                    j += 1
                else:
                    j += 1

    def get_conflicting_neighbors(self, labels: pd.Series) -> pd.Series:
        conflicting_labels = []

        for i in range(0, len(self.indexes)):
            neighbors = self.indexes[i]
            for neighbor in neighbors:
                if labels[neighbor] != labels[i]:
                    conflicting_labels.append(i)

        return pd.Series(conflicting_labels)
