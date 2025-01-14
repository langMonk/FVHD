import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from time import time
from torch.optim import Adam, SGD, Adagrad
from ivhd import IVHD
from knn_graph.graph import Graph

BATCH_SIZE = 60_000
OPTIMIZERS = {"Adam": Adam, "SGD": SGD, "Adagrad": Adagrad, "ForceMethod": None}
EPOCHS = [500, 1000, 1500, 2000]
DEVICES = ["cpu", "cuda"]
N = 3
CREATE_NN = False
NN_S = [2, 5]
RN_S = [1, 2]

dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
X = dataset.data
X = X.reshape(BATCH_SIZE, -1) / 255.
Y = dataset.targets


def create_neighbours(nn, rn):
    RN = torch.randint(0, BATCH_SIZE, (BATCH_SIZE, rn))
    NN = None
    if not CREATE_NN:
        graph = Graph()
        graph.load_from_binary_file("./graph_files/out.bin", nn_count=nn)
        NN = torch.tensor(graph.indexes.astype(np.int32))
    else:
        raise NotImplementedError
    return NN, RN


def make_measurement(nn, rn, epochs, optimizer, optimizer_name, device):
    ivhd = IVHD(nn=nn,
                rn=rn,
                epochs=epochs,
                optimizer=optimizer,
                optimizer_kwargs={"lr": 0.2},
                device=device, verbose=False,
                velocity_limit=False, autoadapt=False)
    x = None
    NN, RN = create_neighbours(nn, rn)
    start = time()
    for _ in range(N):
        x = ivhd.fit_transform(X, NN, RN)
    end = time()
    average_time = (end - start) / N
    print("#"*60)
    print(f"Time for {optimizer_name} NN-{nn} RN-{rn} device-{device} epochs-{epochs}: {average_time}")
    x = x.cpu()

    plt.title(f"{optimizer_name} NN: {nn} RN: {rn} device: {device} epochs: {epochs}")
    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", alpha=0.5, s=1.)
    plt.legend()
    plt.savefig(f"results/{optimizer_name}-{nn}-{rn}-{device}-{epochs}.png")
    plt.close()


if __name__ == '__main__':
    for epochs in EPOCHS:
        for optimizer_name, optimizer in OPTIMIZERS.items():
            for nn in NN_S:
                for rn in RN_S:
                    for device in DEVICES:
                        make_measurement(nn, rn, epochs, optimizer, optimizer_name, device)
