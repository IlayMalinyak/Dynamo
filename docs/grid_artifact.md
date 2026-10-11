# Spot-on-grid discretisation artifact

Script: `scripts/grid_artifact.py` (single constant spot, 5800 K, Kepler band,
one rotation, no noise). Data: `docs/grid_artifact.csv`, figure `docs/grid_artifact.png`.

**Pole-on (i = 0).** The true light curve is flat (the spot circles the line of
sight at fixed mu). Artifact = peak-to-peak flux / spot depth, median over
three spot colatitudes (10, 30, 60 deg):

| rings | grid elements | r = 2 deg | r = 5 deg | r = 10 deg |
|---|---|---|---|---|
| 20 | 969 | 0.146 | 0.043 | 0.030 |
| 30 (current max) | 2218 | 0.156 | 0.047 | 0.015 |
| 40 | 3971 | 0.092 | 0.030 | 0.009 |
| 60 | 9016 | 0.048 | 0.014 | 0.005 |
| 80 | 16096 | 0.040 | 0.010 | 0.003 |

**i = 60 deg.** rms(f_N - f_120) / ptp(f_120) at 30 rings: 3.0% (r = 2), 0.7% (r = 5), 0.2% (r = 10).

Notes
* The artifact is a fixed absolute error per spot, so it matters most where the
  real modulation is small: low inclination and small spots.
* Dynamo picks `n_grid_rings = clip(120 / smallest spot radius, 20, 30)`, so at
  most 30 rings. Spot radii in generated maps span about 1-11 deg, and decaying
  spots shrink further, so pole-on stars carry roughly 5-15% spurious modulation.
* Per-time-step cost scales with the number of grid elements (60 rings is about
  4x and 80 rings about 7x the 30-ring cost).
* Convergence is irregular for small spots (r = 2 deg): the error comes from the
  linear partial-coverage formulas in `nbspectra.calculate_*_coverage`. An exact
  cap-tile overlap or sub-sampling of edge tiles would remove it without a finer grid.
