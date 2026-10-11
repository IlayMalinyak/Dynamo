# Spot-latitude population

The latitude distribution of spots is the population quantity the hierarchical model
should infer, so it is an explicit, saved input of the simulator.

## Model (`lat_model = "gaussian"`, default)

Per star i (columns `Lat Mean`, `Lat Width`, `Lat Drift`, `Rossby`):

* mean   μ_i = mean + rossby_slope · log10(Ro_i / Ro_sun) + N(0, mean_scatter), clipped to [0, mean_max]
* width  σ_i ~ log-uniform(width_min, width_max)
* drift  D_i = drift for butterfly stars, else 0

Each spot: |latitude| ~ N(μ_i(phase), σ_i) truncated to [0°, 90°], random hemisphere,
uniform longitude; with a butterfly cycle μ_i(phase) = μ_i + D_i (0.5 − phase), i.e.
equatorward drift. Spots nested around existing active regions (prob_corr) add a few
degrees of scatter. Ro = P_rot / τ_c with τ_c from Cranmer & Saar 2011 (Ro_sun = 1.97
with the same law). This matches the two-band (mean, width) parametrisation of
`starry_process` (Luger et al. 2021).

| key | default | meaning |
|---|---|---|
| mean | 20 | population mean latitude [deg] |
| mean_scatter | 10 | star-to-star scatter of μ_i [deg] |
| mean_max | 75 | upper clip of μ_i [deg] |
| width_min, width_max | 3, 12 | log-uniform range of σ_i [deg] |
| drift | 20 | cycle drift of the centroid for butterfly stars [deg] |
| rossby_slope | 0 | dμ/dlog10(Ro) [deg/dex]; physical expectation < 0 |
| model | gaussian | `band` restores the original activity-dependent band |

Any value can be `[lo, hi]`: drawn once per table, so many populations with known
hyperparameters can be generated. The resolved values go to `<dataset>/population.json`.
Latitude draws use their own generator: changing the latitude config leaves every
other column of a table unchanged for the same `--base_seed`.

    python -m Dynamo.create_data --dataset_dir pop_A --num_simulations 500 \
        --latitude_config '{"mean": [5, 45], "rossby_slope": [-20, 0]}'

## What the literature says (no measured population prior exists)

* Sun: spots emerge in bands drifting from ~±30° to ~±8° over the cycle; at a given
  time roughly Gaussian with a width of 5-10° (Spörer's law; Hathaway 2015, LRSP 12, 4).
* Theory: faster rotation deflects rising flux poleward, so lower Rossby number means
  higher-latitude spots (Schüssler & Solanki 1992; Işık et al. 2018, A&A 620, A177).
* Doppler imaging of rapid rotators: frequent high-latitude / polar spots
  (Strassmeier 2009, A&ARv 17, 251) — biased to very fast rotators.
* Transit spot crossings: HAT-P-11 low-latitude bands near ±16° (Morris et al. 2017);
  Kepler-63 (young, fast) high-latitude spots (Sanchis-Ojeda et al. 2013).
* `starry_process` (Luger et al. 2021): parametrisation and the demonstration that
  ensembles of light curves constrain (μ, σ); not a measured prior.

The defaults are Sun-like with broad scatter; for recovery tests, draw the
hyperparameters per population.
