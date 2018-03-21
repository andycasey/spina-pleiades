

// Model stellar abundances as a function of cluster abundances and condensation
// temperature for each star

data {
  int<lower=1> S;       // number of stars
  int<lower=1> A;       // number of chemical abundances per star
  vector[A] x;            // condensation temperatures
  vector[A] y[S];         // abundance measurements
  vector[A] yerr[S];      // uncertainties on abundance measurements
  real min_y;
  real max_y;
}

parameters {
  vector[A] c;        // cluster abundance residuals
  vector[S] m;        // slope of abundance with condensation temperature per star
  real<lower=-10, upper=1> ln_f; // natural logarithm of the fraction of underestimated error.
}

transformed parameters {
  vector[A] line[S];
  vector[A] total_error[S];

  for (s in 1:S) {
    line[s, :] = m[s] * x + c;
    //+ c;
    for (a in 1:A)
      total_error[s, a] = (1 + exp(ln_f)) * yerr[s, a];
  }
}

model {
  c ~ normal(0, 1);
  m ~ normal(0, 1e-5);
  for (s in 1:S)
    y[s] ~ normal(line[s], total_error[s]);
}
