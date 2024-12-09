import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ivhd import IVHDGrad
from tests.ivhd.utils import load_mnist


def test_ivhd_interface():
    x = np.random.randn(100, 30)

    ivhd = IVHDGrad()
    x_ivhd = ivhd.fit_transform(x)

    assert isinstance(x_ivhd, np.ndarray), "Expected numpy array"
    assert x_ivhd.shape == (100, 2)


def test_ivhd_performance_on_mnist():
    x, y = load_mnist(10000)

    ivhd = IVHDGrad(
        steps=2000,
        nn=5,
        rn=2,
        re_draw_remote_neighbors=True,
        optimizer_params={"lr": 0.005},
    )
    x_ivhd = ivhd.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_ivhd, y)

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    classification_accuracy = accuracy_score(y_test, y_pred)

    # 72% was experimentally determined to be an average accuracy
    # on this dataset with these parameters.
    # By comparison:
    #   PCA ~ 40%
    #   T-SNE ~ 95%
    # The minimal accuracy threshold to pass the test is set to 65%.
    assert (
        classification_accuracy >= 0.65
    ), "Classification accuracy should be equal or grater than 65%"
