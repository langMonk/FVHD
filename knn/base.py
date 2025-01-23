from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class GraphData:
    indexes: NDArray
    distances: NDArray
