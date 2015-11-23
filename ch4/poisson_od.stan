data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
}

parameters {
  vector[p] beta;
  vector[n] epsilon;
  real<lower=0> sigma;
}

model {
  // priors
  beta ~ normal(0, 3);
  sigma ~ normal(0, 2);
  epsilon ~ normal(0, sigma);

  // likelihood
  y ~ poisson_log(X * beta + epsilon);
}
