data {
  // define the types and names of the data
  int n; // n is an integer
  int<lower=0, upper=1> y[n]; // y is an integer vector with n elements
}

parameters {
  real<lower=0, upper=1> p; // p is a real number between 0 and 1
}

model {
  // define priors
  p ~ beta(1, 1);

  // define likelihood
  y ~ bernoulli(p);
}
