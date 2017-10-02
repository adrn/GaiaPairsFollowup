* The `tgas_apw1.fits` and `tgas_apw2.fits` files were generated to help with
  the mixture modeling for inferring the number of true comoving pairs and
  per-pair probabilities. These files are row-matched and contain all of the
  TGAS data for the stars along with our relative RV measurements. The
  `pair_probs.npy` file is the output of the mixture modeling, and contains the
  per-pair probabilities of being comoving, row-matched to the
  `tgas_apw*.fits` files. For this inference, the field population velocity
  dispersion was set to 25 km/s.
