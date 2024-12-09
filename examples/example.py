from time import time

from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

from ivhd import IVHD, IVHDGrad

dataset = load_digits()
X = dataset["data"]
y = dataset["target"]
X = X / 255.0
# X = X[:30]
# y = y[:30]


ivhd_grad = IVHDGrad(
    steps=1000,
    nn=5,
    rn=2,
    optimizer="adam",
    optimizer_params={"lr": 0.01},
    pos_weight=0.9,
    verbose=True,
    re_draw_remote_neighbors=True,
)
start = time()
X_ivhd_grad = ivhd_grad.fit_transform(X)
stop = time()
print(stop - start)
plt.scatter(X_ivhd_grad[:, 0], X_ivhd_grad[:, 1], c=y, s=2)
plt.show()

if __name__ == "__main__":
    pass
