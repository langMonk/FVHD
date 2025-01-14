import torch
import torchvision
import matplotlib.pyplot as plt

from ivhd.ivdh import IVHD
# from torch import autograd
# import pandas as pd
# import numpy as np
# from knn_graph.faiss_generator import FaissGenerator
# from knn_graph.graph import Graph
# import os
N = None
NN = 10
RN = 1
DATASET_NAME = "emnist"


if __name__ == '__main__':
    # dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    dataset = torchvision.datasets.EMNIST("emnist", split="balanced", train=True, download=True)
    X = dataset.data[:N]
    if not N:
        N = X.shape[0]
    X = X.reshape(N, -1) / 255.
    Y = dataset.targets[:N]

    nn_path = f"./graph_files/{DATASET_NAME}_{X.shape[0]}_{NN}nn.bin"

    ivhd = IVHD(2, NN, RN, c=0.05, eta=0.02, optimizer=None, optimizer_kwargs={"lr": 0.1},
                epochs=3_000, device="mps", velocity_limit=False, autoadapt=False, graph_file=nn_path)

    rn = torch.randint(0, N, (N, RN))


    # if not os.path.exists(nn_path):
    #     # print(f"X shape {X.shape}")
    #     faiss_generator = FaissGenerator(pd.DataFrame(X.numpy()), cosine_metric=False)
    #     faiss_generator.run(nn=NN)
    #     faiss_generator.save_to_binary_file(nn_path)

    # graph = Graph()
    # graph.load_from_binary_file(nn_path, nn_count=NN)
    # nn = torch.tensor(graph.indexes.astype(np.int32))
    # print(f"NN shape: {nn.shape}")


    fig = plt.figure(figsize=(16, 8))
    plt.title(f"{DATASET_NAME} 2d visualization")

    x = ivhd.fit_transform(X).cpu()

    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    plt.legend()
    plt.show()
