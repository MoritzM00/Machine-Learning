import pytest
import numpy as np
from ml.decomposition.pca import PCA
from sklearn.utils._testing import assert_allclose
from sklearn.datasets import load_iris

iris = load_iris()


@pytest.mark.parametrize("n_components", [-1, 0])
def test_invalid_input(n_components):
    with pytest.raises(ValueError):
        PCA(n_components=n_components)


@pytest.mark.parametrize("n_components", range(1, iris.data.shape[1]))
def test_pca_n_components(n_components):
    X = iris.data
    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

    # check that transformed X has the correct shape
    assert X_pca.shape[1] == n_components


def test_pca_check_projection_list():
    # Test that the projection of data is correct
    X = [[1.0, 0.0], [0.0, 1.0]]
    pca = PCA(n_components=1)
    X_trans = pca.fit_transform(X)
    assert X_trans.shape, (2, 1)
    assert_allclose(X_trans.mean(), 0.00, atol=1e-12)
    assert_allclose(X_trans.std(), 0.71, rtol=5e-3)


def test_pca_check_projection():
    # Test that the projection of data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])

    Yt = PCA(n_components=2).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_allclose(np.abs(Yt[0][0]), 1.0, rtol=5e-3)


def test_pca_inverse():
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_allclose(X, Y_inverse, rtol=5e-6)
