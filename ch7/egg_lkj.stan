data {
  // poisson data
  int n;
  int y[n];

  // binomial data
  int n_b;
  int p_index[n_b];
  int k[n_b];
  int x[n_b];
}

parameters {
  matrix[2, n] z;
  vector[2] mu;
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] sigma;
}

transformed parameters {
  matrix[n, 2] alpha;
  vector[n] log_lambda;
  vector[n_b] logit_p;
  alpha <- (diag_pre_multiply(sigma, L) * z)';
  
  for (i in 1:n) log_lambda[i] <- alpha[i, 1];
  log_lambda <- log_lambda + mu[1];
  
  for (i in 1:n_b){
    logit_p[i] <- alpha[p_index[i], 2];
  }
  logit_p <- logit_p + mu[2];
}

model {
  // priors
  mu ~ normal(0, 2);
  sigma ~ normal(0, 2);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ normal(0, 1);

  // likelihood
  y ~ poisson_log(log_lambda);
  x ~ binomial_logit(k, logit_p);
}

generated quantities {
  matrix[2, 2] Rho;
  
  Rho <- multiply_lower_tri_self_transpose(L);
}