import numpy as np
import os
import time
import make_data as md
from glum import GeneralizedLinearRegressor as GLR

data_prefix = md.data_prefix

def get_data(n, p):
    make_file_path = lambda suff: \
        os.path.join(data_prefix, ''.join([str(n), '_', str(p), '_', suff, '.csv']))
    X_file = make_file_path('X')
    y_file = make_file_path('y')
    return np.genfromtxt(X_file, delimiter=','), np.genfromtxt(y_file, delimiter=',')

def timer(X, y, glr):
    start = time.time()
    glr_fit = glr.fit(X, y)
    end = time.time()
    return glr_fit, end-start

n = 100000
p = 100
X, y = get_data(n, p)
glr = GLR(l1_ratio=1,
          family='normal',
          gradient_tol=1e-7,
          scale_predictors=False,
          min_alpha_ratio=0.01 if n < p else 1e-4,
          solver='irls-cd',
          n_alphas=62,
          alpha_search=True,
          warm_start=True
          )
glr_fit, elapsed = timer(X, y, glr)
print("Coef:\n", glr_fit.coef_)
print("Intercept: ", glr_fit.intercept_)
print("N_iter: ", glr_fit.n_iter_)
print("Elapsed: ", elapsed)
