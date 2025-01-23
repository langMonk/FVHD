import ssl

import pytest
import torch
from torchvision import datasets

from fvhd import FVHD


def setup_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


setup_ssl()


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize(
    "optimizer", [None, torch.optim.Adam, torch.optim.SGD, torch.optim.Adagrad]
)
def test_basic_fvhd(device, optimizer):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    if not torch.backends.mps.is_available() and device == "mps":
        pytest.skip("MPS not available")

    NN = torch.tensor([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    RN = torch.tensor([[3], [4], [3], [0], [1], [2]])
    X = torch.rand((6, 50))

    fvhd = FVHD(
        n_components=2,
        nn=2,
        rn=1,
        c=0.3,
        optimizer=optimizer,
        optimizer_kwargs={"lr": 0.1} if optimizer else None,
        epochs=300,
        eta=0.1,
        device=device,
    )

    embeddings = torch.tensor(fvhd.fit_transform(X, graph=create_mock_graph(NN)))
    assert embeddings.shape == (6, 2)

    embeddings = embeddings.reshape(6, 1, 2)
    nn_diffs = embeddings - torch.index_select(embeddings, 0, NN.reshape(-1)).reshape(
        6, -1, 2
    )
    rn_diffs = embeddings - torch.index_select(embeddings, 0, RN.reshape(-1)).reshape(
        6, -1, 2
    )

    nn_dist = torch.sqrt(torch.sum(nn_diffs**2 + 1e-8, dim=-1))
    rn_dist = torch.sqrt(torch.sum(rn_diffs**2 + 1e-8, dim=-1))

    assert torch.mean(nn_dist).item() < 0.5
    assert abs(torch.mean(rn_dist).item() - 1.0) < 0.5


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("optimizer", [None, torch.optim.Adam, torch.optim.SGD])
@pytest.mark.parametrize("n_samples", [100, 500])
@pytest.mark.parametrize("nn_count", [2, 5])
@pytest.mark.parametrize("rn_count", [1, 2])
def test_mnist_fvhd(device, optimizer, n_samples, nn_count, rn_count):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    if not torch.backends.mps.is_available() and device == "mps":
        pytest.skip("MPS not available")

    torch.manual_seed(42)

    dataset = datasets.MNIST("mnist", train=True, download=True)
    X = dataset.data[:n_samples].reshape(n_samples, -1) / 255.0

    from sklearn.decomposition import PCA

    pca = PCA(n_components=50)
    X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

    distances = torch.cdist(X, X)
    _, nn = torch.topk(distances, nn_count + 1, dim=-1, largest=False)
    NN = nn[:, 1:]
    RN = torch.randint(0, n_samples, (n_samples, rn_count))

    fvhd = FVHD(
        n_components=2,
        nn=nn_count,
        rn=rn_count,
        c=0.4,
        optimizer=optimizer,
        optimizer_kwargs={"lr": 0.1} if optimizer else None,
        epochs=600,
        eta=0.2,
        device=device,
        velocity_limit=True,
    )

    embeddings = fvhd.fit_transform(X, graph=create_mock_graph(NN))
    assert embeddings.shape == (n_samples, 2)

    embeddings = torch.tensor(embeddings).reshape(n_samples, 1, 2)

    nn_tensor = torch.tensor(NN)
    rn_tensor = torch.tensor(RN)

    nn_diffs = embeddings - torch.index_select(
        embeddings, 0, nn_tensor.reshape(-1)
    ).reshape(n_samples, -1, 2)
    rn_diffs = embeddings - torch.index_select(
        embeddings, 0, rn_tensor.reshape(-1)
    ).reshape(n_samples, -1, 2)

    nn_dist = torch.sqrt(torch.sum(nn_diffs**2 + 1e-8, dim=-1))
    rn_dist = torch.sqrt(torch.sum(rn_diffs**2 + 1e-8, dim=-1))

    assert torch.mean(nn_dist).item() < 0.5
    assert abs(torch.mean(rn_dist).item() - 1.0) < 0.5


def create_mock_graph(nn_indices):
    from knn.base import GraphData
    from knn.graph import Graph

    distances = torch.ones_like(nn_indices, dtype=torch.float32)
    return Graph(GraphData(indexes=nn_indices.numpy(), distances=distances.numpy()))
