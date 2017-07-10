functions {
    real spacedensity_lpdf(real r, real r_lim){
        real ln_prob;
        if (r < r_lim)
            ln_prob = log(3/r_lim^3) + 2*log(r);
        else
            ln_prob = negative_infinity();
        return ln_prob;
    }

    real separation_lpdf(real s, real alpha, real a, real b){
        real ln_prob;
        real A;

        if (s < a)
            ln_prob = negative_infinity();

        else if (s > b)
            ln_prob = negative_infinity();

        else {
            A = (alpha + 1) / (b^(alpha+1) - a^(alpha+1));
            ln_prob = log(A) + alpha*log(s);
        }

        return ln_prob;
    }
}

data {
    int<lower=1> N; // total number of comoving pairs

    // Units for data and Cov below should be:
    //     ra [radian]
    //     dec [radian]
    //     parallax [mas]
    vector[3] data1[N]; // star 1: ra [rad], dec [rad], parallax [mas]
    vector[3] data2[N]; // star 2: ra [rad], dec [rad], parallax [mas]

    vector[N] plx_err1; // parallax error [mas]
    vector[N] plx_err2; // parallax error [mas]
}

transformed data {
    vector[3] u1[N];
    vector[3] u2[N];
    real a; // minimum separation
    real b; // maximum separation
    real r_lim; // maximum distance
    real Q; // probability that d1 > d2
    Q = 0.5;

    for(n in 1: N) {
        u1[n,1] = cos(data1[n,1]) * cos(data1[n,2]);
        u1[n,2] = sin(data1[n,1]) * cos(data1[n,2]);
        u1[n,3] = sin(data1[n,2]);

        u2[n,1] = cos(data2[n,1]) * cos(data2[n,2]);
        u2[n,2] = sin(data2[n,1]) * cos(data2[n,2]);
        u2[n,3] = sin(data2[n,2]);
    }

    r_lim = 1000.; // pc
    a = 0.01; // pc
    b = 100.; // pc
}

parameters {
    vector<lower=0, upper=r_lim>[N] d1; // distance to star 1
    vector[N] dr; // difference in distance, d1-d2
    real<lower=-10, upper=10> alpha;
}

transformed parameters {
    vector[N] true_plx1;
    vector[N] true_plx2;
    vector<lower=0>[N] s;
    vector<lower=0>[N] d2;

    for(n in 1: N) {
        d2[n] = d1[n] + dr[n];

        if (d2[n] < 0)
            reject("BAD SAMPLE");

        s[n] = sqrt( (d1[n]*u1[n,1] - d2[n]*u2[n,1])^2 +
                     (d1[n]*u1[n,2] - d2[n]*u2[n,2])^2 +
                     (d1[n]*u1[n,3] - d2[n]*u2[n,3])^2 );

        true_plx1[n] = 1000. / d1[n];
        true_plx2[n] = 1000. / d2[n];
    }
}

model {
    vector[2] terms;

    alpha ~ uniform(-10, 10); // power law index

    for(n in 1: N) {
        data1[n,3] ~ normal(true_plx1, plx_err1); // Gaussian error on parallax
        d1[n] ~ spacedensity(r_lim); // uniform space-density prior on distance

        data2[n,3] ~ normal(true_plx2, plx_err2); // Gaussian error on parallax
        s[n] ~ separation(alpha, a, b);
    }

}
