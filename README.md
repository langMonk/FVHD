# FVHD: Fast Visualization of High-Dimensional Data

FVHD is a Python library for efficient visualization of high-dimensional data using force-directed graph layout algorithms. It provides an implementation of high-dimensional data visualization with neighbor-based force calculations.

## Features

- Fast neighbor search using scikit-learn's NearestNeighbors
- Force-directed graph layout optimization
- Support for both optimizer-based and force-directed methods
- Automatic parameter adaptation
- Efficient binary graph storage format

## Installation

FVHD requires Python 3.12+. We recommend using [uv](https://docs.astral.sh/uv/) for dependency management:

1.  **Install uv** (if missing):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

3.  **Run scripts**:
    ```bash
    uv run python your_script.py
    ```

## Quick Start

```python
import torch
import pandas as pd
from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator

X = torch.rand(1000, 784)
df = pd.DataFrame(X.numpy())

config = NeighborConfig(metric="euclidean")
generator = NeighborGenerator(df=df, config=config)
graph_knn, graph_mutual = generator.run(nn=5)

fvhd = FVHD(
    n_components=2,
    nn=5,
    rn=2,
    c=0.1,
    eta=0.2,
    epochs=3000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    velocity_limit=True,
    autoadapt=True
)

embeddings = fvhd.fit_transform(X, graph=graph_knn)
```

## Example with MNIST

```python
from fvhd import FVHD

fvhd = FVHD(
    n_components=2,
    nn=5,
    rn=2,
    c=0.005,
    eta=0.003,
    epochs=3000,
    device="cuda"
)

embeddings = fvhd.fit_transform(X)
```

## Parameters

- `n_components`: Output dimensionality (default: 2)
- `nn`: Number of nearest neighbors (default: 2)
- `rn`: Number of random neighbors (default: 1)
- `c`: Repulsion strength coefficient (default: 0.1)
- `eta`: Learning rate (default: 0.1)
- `epochs`: Number of optimization epochs (default: 200)
- `device`: Computation device ("cpu", "cuda", "mps")
- `autoadapt`: Enable automatic learning rate adaptation
- `velocity_limit`: Enable velocity limiting for stability

## Citation

If you use this software in your research, please cite:

```
@misc{fvhd2025,
  author = {Minch, Bartosz, Ręka, Filip and Dzwinel, Witold},
  title = {FVHD: Fast Visualization of High-Dimensional Data},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/langMonk/fvhd}
}
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
