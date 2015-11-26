data {
   int<lower=0> nsite;
   int<lower=0> nsurvey;
   int<lower=0,upper=1> y[nsite,nsurvey];
}
parameters {
   real<lower=0,upper=1> psi;
   real<lower=0,upper=1> p;
}
model {
   for (i in 1:nsite) {
     if (sum(y[i]) > 0)
       // species was observed: it is there
       increment_log_prob(log(psi) + bernoulli_log(y[i],p));
     else
       // it may or may not be there
       increment_log_prob(log_sum_exp(log(psi) + bernoulli_log(y[i],p),
                                      log1m(psi)));
   }
}
