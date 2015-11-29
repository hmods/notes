data {
  int n;
  vector[n] y;

  // covariate
  int tmax;
  vector<lower=0, upper=tmax>[n] t;
  
  // indices
  int n_subject;
  int<lower=1, upper=n_subject> subject[n];

}

parameters {
  matrix[2, n_subject] z;
  vector[2] mu;
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] sigma;
  real<lower=0> sigma_y;
}

transformed parameters {
  matrix[n_subject, 2] alpha;
  vector[n] mu_y;
  
  alpha <- rep_matrix(mu', n_subject) 
              + (diag_pre_multiply(sigma, L) * z)';
  
  for (i in 1:n) mu_y[i] <- alpha[subject[i], 1] + alpha[subject[i], 2] * t[i];
}

model {
  to_vector(z) ~ normal(0, 1);
  sigma ~ cauchy(0, 5);
  sigma_y ~ cauchy(0, 5);
  L ~ lkj_corr_cholesky(2);
  mu ~ normal(200, 50);
  
  y ~ normal(mu_y, sigma_y);
}

generated quantities {
  matrix[2, 2] Rho;
  Rho <- multiply_lower_tri_self_transpose(L);
}