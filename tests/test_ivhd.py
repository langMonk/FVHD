import pytest
import torch
import torchvision

from fvhd import FVHD

dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "optimizer", [None, torch.optim.Adam, torch.optim.SGD, torch.optim.Adagrad]
)
def test_sanity_ivhd(device, optimizer):
    NN = torch.tensor([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    RN = torch.tensor([[3], [4], [3], [0], [1], [2]])
    X = torch.zeros((6, 3))
    ivhd = FVHD(
        2,
        2,
        1,
        0.3,
        optimizer=optimizer,
        optimizer_kwargs={"lr": 0.1},
        epochs=300,
        eta=0.1,
        device=device,
        verbose=True,
    )

    x_2d = ivhd.fit_transform(X=X, NN=NN, RN=RN)
    print(x_2d)
    assert x_2d.shape == (6, 2)
    assert str(x_2d.device) == device

    x_2d = x_2d.reshape(6, 1, 2)
    x_2d = x_2d.cpu()
    nn_diffs = x_2d - torch.index_select(x_2d, 0, NN.reshape(-1)).reshape(
        X.shape[0], -1, 2
    )
    rn_diffs = x_2d - torch.index_select(x_2d, 0, RN.reshape(-1)).reshape(
        X.shape[0], -1, 2
    )
    nn_dist = torch.sqrt(torch.sum((nn_diffs + 1e-8) ** 2, dim=-1, keepdim=True))
    rn_dist = torch.sqrt(torch.sum((rn_diffs + 1e-8) ** 2, dim=-1, keepdim=True))

    assert torch.allclose(nn_dist, torch.tensor(0.0), rtol=1e-3, atol=1e-3)
    assert torch.allclose(rn_dist, torch.tensor(1.0), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "optimizer", [None, torch.optim.Adam, torch.optim.SGD, torch.optim.Adagrad]
)
@pytest.mark.parametrize("N", [150, 100])
@pytest.mark.parametrize("NN", [2, 5])
@pytest.mark.parametrize("RN", [1, 2])
def test_mnist_part_ivhd(device, optimizer, N, NN, RN):
    torch.manual_seed(0)
    X = dataset.data[:N]
    X = X.reshape(N, -1) / 255.0
    distances = torch.zeros(N, N)
    for i in range(N):
        distances[i] = torch.sum((X[i] - X) ** 2, dim=-1)
    _, nn = torch.topk(distances, NN + 1, dim=-1, largest=False)
    NN_tensor = nn[:, 1:]
    RN_tensor = torch.randint(0, N, (N, RN))

    ivhd = FVHD(
        2,
        NN,
        RN,
        0.4,
        optimizer=optimizer,
        optimizer_kwargs={"lr": 0.1},
        epochs=600,
        eta=0.2,
        device=device,
        velocity_limit=True,
        verbose=True,
    )

    x_2d = ivhd.fit_transform(X=X, NN=NN_tensor, RN=RN_tensor)

    assert x_2d.shape == (N, 2)
    assert str(x_2d.device) == device

    x_2d = x_2d.reshape(N, 1, 2)
    x_2d = x_2d.cpu()
    nn_diffs = x_2d - torch.index_select(x_2d, 0, NN_tensor.reshape(-1)).reshape(
        N, -1, 2
    )
    rn_diffs = x_2d - torch.index_select(x_2d, 0, RN_tensor.reshape(-1)).reshape(
        N, -1, 2
    )
    nn_dist = torch.sqrt(torch.sum((nn_diffs + 1e-8) ** 2, dim=-1, keepdim=True))
    rn_dist = torch.sqrt(torch.sum((rn_diffs + 1e-8) ** 2, dim=-1, keepdim=True))

    print()
    print(torch.mean(nn_dist))
    print(torch.mean(rn_dist))
    # we will use mean because some data may be unable to fit
    assert torch.allclose(torch.mean(nn_dist), torch.tensor(0.0), rtol=3e-1, atol=3e-1)
    assert torch.allclose(torch.mean(rn_dist), torch.tensor(1.0), rtol=3e-1, atol=3e-1)
