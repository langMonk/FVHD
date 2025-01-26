# FVHD: Fast Visualization of High-Dimensional Data

FVHD is a Python library for efficient visualization of high-dimensional data using force-directed graph layout algorithms. It provides an implementation of high-dimensional data visualization with neighbor-based force calculations.

## Features

- Fast neighbor search using scikit-learn's NearestNeighbors
- Force-directed graph layout optimization
- Support for both optimizer-based and force-directed methods
- Automatic parameter adaptation
- Built-in support for MNIST and EMNIST datasets
- Efficient binary graph storage format

## Installation

FVHD requires Python 3.12 and can be installed using Poetry:

```bash
poetry install
```

## Quick Start

```python
import torch
from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator

# Load your data as a torch.Tensor
X = torch.rand(1000, 784)  # Example: 1000 samples of 784 dimensions

# Create nearest neighbors graph
config = NeighborConfig(metric="euclidean")
generator = NeighborGenerator(df=df, config=config)
graph = generator.run(nn=5)

# Initialize FVHD
fvhd = FVHD(
    n_components=2,  # Output dimensionality
    nn=5,           # Number of nearest neighbors
    rn=2,           # Number of random neighbors
    c=0.1,        # Repulsion strength
    eta=0.2,      # Learning rate
    epochs=3000,
    device="cuda",  # Use GPU if available
    velocity_limit=True,
    autoadapt=True
)

# Generate 2D embeddings
embeddings = fvhd.fit_transform(X, graph)
```

## Example with MNIST

```python
from main import load_dataset, create_or_load_graph, visualize_embeddings

# Load MNIST dataset
X, Y = load_dataset("mnist")

# Create nearest neighbors graph
graph = create_or_load_graph(X, nn=5)

# Initialize and run FVHD
fvhd = FVHD(
    n_components=2,
    nn=5,
    rn=2,
    c=0.005,
    eta=0.003,
    epochs=3000,
    device="cuda"
)

# Generate and visualize embeddings
embeddings = fvhd.fit_transform(X, graph)
visualize_embeddings(embeddings, Y, "mnist")
```

## Parameters

- `n_components`: Output dimensionality (default: 2)
- `nn`: Number of nearest neighbors (default: 2)
- `rn`: Number of random neighbors (default: 1)
- `c`: Repulsion strength coefficient (default: 0.1)
- `eta`: Learning rate (default: 0.1)
- `epochs`: Number of training epochs (default: 200)
- `device`: Computation device ("cpu" or "cuda")
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
