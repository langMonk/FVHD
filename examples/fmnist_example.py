import ssl

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml

from ivhd import IVHD, IVHDGrad

ssl._create_default_https_context = ssl._create_unverified_context


def load_dataset(dataset_name):
    mnist = fetch_openml(dataset_name, version=1)
    X, y = mnist["data"], mnist["target"].astype(np.int32)

    return X, y


def visualise(X, Y, dataset_name, algorithm_name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=Y, palette="tab10", legend="full", alpha=0.6
    )
    plt.title(f"{algorithm_name} of {dataset_name} Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig(f"{algorithm_name}_on_{dataset_name}.png")
    plt.show()


def perform_ivhd(X, n_components, nn, rn, c, lambda_, simulation_steps):
    ivhd = IVHD(
        n_components=n_components,
        nn=nn,
        rn=rn,
        c=c,
        simulation_steps=simulation_steps,
        lambda_=lambda_,
    )
    X_ivhd = ivhd.fit_transform(X)
    return X_ivhd


def perform_grad_ivhd(
    X, n_components, nn, rn, pos_weight, optimizer, optimizer_params, simulation_steps
):
    ivhd_grad = IVHDGrad(
        n_components=n_components,
        steps=simulation_steps,
        nn=nn,
        rn=rn,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        pos_weight=pos_weight,
        re_draw_remote_neighbors=True,
    )
    X_ivhd_grad = ivhd_grad.fit_transform(X)
    return X_ivhd_grad


if __name__ == "__main__":
    X, y = load_dataset("Fashion-MNIST")

    X_ivhd = perform_ivhd(
        X, n_components=2, nn=5, rn=5, c=0.1, simulation_steps=1500, lambda_=0.95
    )
    visualise(X_ivhd, y, "Fashion-MNIST", "IVHD")

    X_grad_ivhd = perform_grad_ivhd(
        X,
        n_components=2,
        nn=5,
        rn=5,
        pos_weight=0.9,
        optimizer="adam",
        optimizer_params={"lr": 0.01},
        simulation_steps=1500,
    )
    visualise(X_grad_ivhd, y, "Fashion-MNIST", "Gradient IVHD")
