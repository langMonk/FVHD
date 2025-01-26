import ssl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator


def setup_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def load_dataset(name: str, n_samples: Optional[int] = None):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    elif name == "emnist":
        dataset = torchvision.datasets.EMNIST(
            "emnist", split="balanced", train=True, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = dataset.data[:n_samples]
    N = len(X) if n_samples is None else n_samples
    X = X.reshape(N, -1) / 255.0

    from sklearn.decomposition import PCA

    pca = PCA(n_components=50)
    X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

    Y = dataset.targets[:n_samples]
    return X, Y


def create_or_load_graph(X: torch.Tensor, nn: int) -> Graph:
    config = NeighborConfig(metric="euclidean")
    df = pd.DataFrame(X.numpy())
    generator = NeighborGenerator(df=df, config=config)
    graph = generator.run(nn=nn)
    return graph


def visualize_embeddings(x: np.ndarray, y: torch.Tensor, dataset_name: str):
    plt.switch_backend("TkAgg")
    plt.figure(figsize=(8, 8))
    plt.title(f"{dataset_name} 2d visualization")

    y = y.numpy()
    for i in range(10):
        points = x[y == i]
        plt.scatter(
            points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    setup_ssl()

    DATASET_NAME = "mnist"

    X, Y = load_dataset(DATASET_NAME)
    graph = create_or_load_graph(X, 5)

    fvhd = FVHD(
        n_components=2,
        nn=5,
        rn=2,
        c=0.02,
        eta=0.2,
        optimizer=None,
        optimizer_kwargs={"lr": 0.1},
        epochs=2000,
        device="mps",
        velocity_limit=True,
        autoadapt=True,
    )

    embeddings = fvhd.fit_transform(X, graph)
    visualize_embeddings(embeddings, Y, DATASET_NAME)
