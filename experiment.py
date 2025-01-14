import argparse
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from torchvision.datasets import EMNIST, MNIST
from torch.optim import Adam, Adagrad


def run(
    dataset:Literal["mnist", "emnist", "rcv", "amazon"],
    model:Literal["ivhd", "umap", "pacmap", "tsne", "trimap"],
    interactive=False,
    device : Literal['cuda', 'cpu', 'auto', 'mps']="auto",
    graph_file="",
    save_output=False,
):
    assert device in {"cpu", "cuda", "mps", "auto"}, f"Invalid device: {device}. Allowed values are 'cpu', 'cuda', 'mps' or 'auto'."
    if device == "auto":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    match dataset:
        case "mnist":
            data = MNIST("mnist", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.0
            Y = data.targets[:N]
            pca = PCA(n_components=50)
            X = torch.Tensor(pca.fit_transform(X))
        case "emnist":
            data = EMNIST("emnist", split="balanced", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.0
            Y = data.targets[:N]
        case "20ng":
            newsgroups = fetch_20newsgroups(
                data_home="20ng", subset="all", remove=("headers", "footers", "quotes")
            )
            posts = newsgroups.data
            Y = torch.Tensor(newsgroups.target)
            vectorizer = TfidfVectorizer(max_features=1000)
            tmp = vectorizer.fit_transform(posts).toarray()
            scaler = MinMaxScaler()
            tmp = scaler.fit_transform(tmp)
            pca = PCA(n_components=50)
            X = torch.Tensor(pca.fit_transform(tmp))
        case "higgs":
            if not Path("./HIGGS.csv").is_file():
                import subprocess

                subprocess.run(
                    ["wget", "https://archive.ics.uci.edu/static/public/280/higgs.zip"]
                )
                subprocess.run(["unzip", "higgs.zip"])
                subprocess.run(["gzip", "-d", "HIGGS.csv.gz"])
                pass
            X = np.loadtxt("HIGGS.csv", delimiter=",")
            Y = torch.Tensor(X[:, 0])
            X = torch.Tensor(X[:, 1:])
        case _:
            raise ValueError("only mnist, emnist, rcv1 and amazon are supported")

    match model:
        case "ivhd":
            from ivhd.ivdh import IVHD
            model_dr = IVHD(
                2,
                nn=2,
                rn=1,
                c=0.05,
                eta=0.02,
                optimizer=None,
                # optimizer=Adam,
                optimizer_kwargs={"lr": 0.1},
                epochs=8_000,
                device=device,
                velocity_limit=False,
                autoadapt=False,
                graph_file="./graph_files/graph.bin"
            )
        case "pacmap":
            if device == "cuda":
                from parampacmap import ParamPaCMAP
                model_dr = ParamPaCMAP(verbose=True)
            else:
                from pacmap import PaCMAP
                model_dr = PaCMAP()
        case "tsne":
            if device == "cuda":
                from cuml import TSNE
            else:
                from openTSNE import TSNE
            model_dr = TSNE(verbose=True)
            X = X.numpy()
        case "umap":
            if device == "cuda":
                from cuml import UMAP
            else:
                from umap import UMAP
            model_dr = UMAP(verbose=True)
            X = X.numpy()
        case "trimap":
            if device == "cuda":
                from models.trimap import TRIMAP
            else:
                from trimap import TRIMAP
                X = X.numpy()
            model_dr = TRIMAP(verbose=True)
        case _:
            raise ValueError("Only support ivhd, pacmap, tsne, umap")

    # maybe we should use default timer or sth like that to be more accurate
    start = time()
    x = model_dr.fit_transform(X)
    end = time()
    print(end - start)

    save_dir = f'./experiments/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{model}_{dataset}_{device}'
    os.makedirs(exist_ok=True, name=save_dir)

    if save_output:
        data = np.hstack((x, Y.numpy().reshape(-1, 1)))
        np.savetxt(
            str(Path(save_dir, "reduced.txt")),
            data,
            delimiter=",",
            header="Feature1,Feature2,Label",
            comments="",
            fmt="%.6f",
        )

    with open(str(Path(save_dir, "timing.txt")), "w") as f:
        f.write(str(end - start))

    fig = plt.figure(figsize=(16, 8))
    plt.title(f"{dataset} 2d visualization")

    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=Y,
        s=2,
        palette=sns.color_palette("tab10", torch.unique(Y).size(0)),
    )
    plt.legend()
    if interactive:
        plt.show()
    else:
        plt.savefig(str(Path(save_dir, "image.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the dataset and model configuration."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "emnist", "20ng", "higgs"],
        required=True,
        help="Specify the dataset to use. Choices are: 'mnist', 'emnist', '20ng', 'higgs'.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["ivhd", "umap", "pacmap", "tsne", "trimap"],
        required=True,
        help="Specify the model to use. Choices are: 'ivhd', 'umap', 'pacmap', 'tsne', 'trimap'.",
    )

    parser.add_argument("--graph", type=str, dest="graph")

    parser.add_argument(
        "--interactive",
        action="store_true",
        dest="interactive",
        help="Anable interactive mode. Save the plot instead of displaying it.",
    )

    parser.add_argument("--save-output", action="store_true", dest="save_output")

    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", dest='device', help="Device to use: 'cpu', 'cuda', 'mps' or 'auto'.")

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        model=args.model,
        interactive=args.interactive,
        graph_file=args.graph,
        save_output=args.save_output,
        device=args.device
    )
