

// Simple cluster abundance model with some intrinsic scatter

data {
  int<lower=1> S; // number of stars with measured abundances
  real y[S];      // measurements
  real yerr[S];   // measurement errors
}

parameters {
  real abundance;       // cluster abundance
  real scatter;         // intrinsic scatter in abundances
}

transformed parameters {
  real sigma[S];
  real intrinsic_variance;

  intrinsic_variance = pow(scatter, 2);
  for (s in 1:S)
    sigma[s] = pow(pow(yerr[s], 2) + intrinsic_variance, 0.5);
}

model {
  y ~ normal(abundance, sigma);
}

generated quantities {
  real intrinsic_scatter;
  intrinsic_scatter = pow(intrinsic_variance, 0.5);
}