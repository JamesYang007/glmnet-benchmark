import numpy as np
import os

data_prefix = 'data'

class GaussianData(object):
    def __init__(self, n, p):
        self.n = n
        self.p = p
    def make_X(self):
        return np.random.normal(size=(self.n, self.p))
    def make_beta(self, sparsity=0.5):
        nonzeros = np.random.binomial(1, sparsity, size=self.p)
        n_nonzero = np.sum(nonzeros == 1)
        nonzeros[nonzeros == 1] = np.random.normal(size=n_nonzero)
        return nonzeros
    def make_y(self, X, beta):
        return X.dot(beta) + np.random.normal(size=self.n)

def make_file_path(n, p, suff):
    return os.path.join(data_prefix, ''.join([str(n), '_', str(p), '_', suff, '.csv']))

if __name__ == '__main__':
    n = 100000
    p = 100
    gd = GaussianData(n, p)
    X, beta = gd.make_X(), gd.make_beta()
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = gd.make_y(X, beta)
    np.savetxt(make_file_path(n, p, 'X'), X, delimiter=',')
    np.savetxt(make_file_path(n, p, 'y'), y, delimiter=',')
