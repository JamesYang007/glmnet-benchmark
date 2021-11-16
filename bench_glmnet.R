library(glmnet)
library(microbenchmark)

data_prefix = 'data'

get_data = function(n, p) {
    make_file_path = function(suff) {
        paste(data_prefix, '/', n, '_', p, '_', suff, '.csv', sep='')
    }
    X_file = make_file_path('X')
    y_file = make_file_path('y')
    list(X=read.csv(X_file, header=F), y=read.csv(y_file, header=F))
}

timer = function(X, y) {
    time.out = microbenchmark(glmnet(X, y, family='gaussian', tol=1e-14, standardize=F, lambda=0.0074), times=1L, unit='s')
    glmnet.out = glmnet(X, y, family='gaussian', tol=1e-14, standardize=F, lambda=0.0074)
    list(out=glmnet.out, elapsed=summary(time.out)$mean)
}

n = 100000L
p = 100L
dat = get_data(n, p)
X = as.matrix(dat$X)
y = as.numeric(dat$y[,1])
out = timer(X, y)
glmnet.fit = out$out
elapsed = out$elapsed
print("Coef:\n")
print(glmnet.fit$beta)
print(paste("Intercept:", glmnet.fit$a0))
print(paste("N_iter:", glmnet.fit$npasses))
print(paste("Elapsed:", elapsed))
