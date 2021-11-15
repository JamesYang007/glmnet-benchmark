# Glmnet Benchmark

## Glmnet vs. Glum

Before running the benchmark, run the following:
```bash
mkdir -p data
```

This benchmark tests a simple case where the data matrix `X` has rows sampled from `N(0, I_p)`
and `y = X * beta + eps` where `eps ~ N(0, I_n)` and `beta` has some true sparsity (i.e. some proportion of the entries are 0).
See [make_data.py](make_data.py) for details.
Note that we standardize the columns of `X`.

The files `bench_glum.py` and `bench_glmnet.R` provide two functions `get_data` and `timer`.
The `get_data` function simply reads the generated data, which is stored by default in `data/`,
and returns `(X,y)` pair.
__Please read carefully the settings of the lasso solver__.
A couple notes:
- The current setting benchmarks __pathwise solution__.
- We supply the character string `"gaussian"` to `glmnet` to invoke the C++ routine.
- Both methods __do not__ standardize `X`.
- We only tested the lasso (`l1_ratio=1` in `glum` and `alpha=1` (default) in `glmnet`).
- `min_alpha_ratio` in `glum` (or `lambda.min.ratio` in `glmnet`) was fixed to be the same behavior as the default behavior in `glmnet`.
  This way, the regularization path is exactly the same.
- We fix the solver to be `irls-cd` for `glum`, just to make it clear. 
  According to the documentation, it uses `irls-cd` anyways.
- Because of early-stopping rules for `glmnet`, 
  the model may not be fit on _all_ the number of points on the regularization path.
  To make it an absolutely fair benchmark, we also changed the `n_alphas` parameter in `glum` to match
  the total number of points realized after fitting with `glmnet`.
- Both methods use warm-starts.
- The biggest challenge in making this a fair benchmark is choosing the threshold.
  Glum seems to sum the absolute change in `beta_i` values after each coordinate descent
  and checks if that sum is below the threshold.
  Glmnet takes the maximum of the squared difference in `beta_i` after each coordinate descent
  and checks if that is below the threshold.
  This is why the tolerance for `glmnet` is that of `glum` squared.

The workflow is as follows:
- Go in each of the files `bench_glum.py`, `bench_glmnet.R`, `make_data.py` and change `n, p`.
- If you haven't created data for the current value of `n, p`, run `python make_data.py`.
- Run the benchmarks separately:
```bash
python bench_glum.py
Rscript bench_glmnet.R
```
- Double-check that the outputs match (more or less), the number of iterations has not reached the max,
  and compare the elapsed times.
