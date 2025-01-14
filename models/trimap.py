import jax_trimap
import jax.random as random
import numpy as np

class TRIMAP:
    def __init__(self, distance='euclidean', verbose=False):
        self.key = random.PRNGKey(42)
        self.distance = distance
        self.verbose = verbose
    
    def fit_transform(self, x,):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return jax_trimap.transform(self.key, x, distance=self.distance, verbose=self.verbose)
