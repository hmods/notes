data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
  int k[n];
}

parameters {
  vector[p] beta;
}

model {
  beta ~ normal(0, 2);
  y ~ binomial_logit(k, X * beta);
}
