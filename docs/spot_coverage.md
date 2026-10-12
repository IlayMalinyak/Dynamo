# Spot coverage vs Rossby number (replaces the 500-spot cap)

## Problem

The number of spots emerged over a run was `0.1 sqrt(ar) ndays`, clipped to [20, 500]
(`spots.py`, comment "arbitrary"). In the faculae smoke run, 24-40% of spotted stars sat at
the 500 cap (every star with ar > 25) and produced a second peak in the amplitude
distribution (docs/faculae_smoke.md). Expressed as coverage, the old rule gives a median
0.86% of the surface and only a factor ~3 between saturated (1.4%) and solar-Rossby (0.5%)
stars. That is much flatter than the measured relation below.

## What the literature gives

| quantity | value | source |
|---|---|---|
| sunspot coverage at solar maximum | 0.28% of the disk (0.003% at minimum) | Shapiro et al. 2014, A&A, doi:10.1051/0004-6361/201323086 (arXiv:1406.2383) |
| sunspot coverage, cycle 23 maximum | ~0.3% of the visible disk (2002) | Watson, Fletcher & Marshall 2011, A&A 533, A14 (arXiv:1108.4285) |
| spectroscopic f_spot vs Ro, Pleiades (APOGEE, 2-temperature) | f = 0.0634 Ro^-0.881 for log Ro >= -0.729; 0.278 below (eq. 5); or 0.248 plateau, -0.569 log Ro - 0.137 (eq. 4) | Cao & Pinsonneault 2022, MNRAS 517, 2165 (arXiv:2209.10549) |
| scatter at saturation | 0.059-0.066 (absolute), ~21% relative | same |
| solar Rossby number in their convention | log Ro_sun = 0.39 (SPOTS tau_CZ) | same |
| f_spot, M67 (solar age) | GK dwarfs 0.030 +- 0.008; solar analogues (5700-5900 K) 0.020 +- 0.011; giants 0.003 +- 0.002 | same |
| TiO f_spot, Pleiades | rises with decreasing Ro, saturates at 40-50% below Ro ~ 0.1 (Wright 2011 tau) | Fang et al. 2016, MNRAS 463, 2494 (arXiv:1608.05452) |
| f_spot vs age, Kepler/K2 clusters | f ~ 0.014 (t/Gyr)^-0.37; f ~ 0.02 SmAmp[%]^0.74 | Morris 2020, ApJ, doi:10.3847/1538-4357/ab79a0 (arXiv:2002.09135) |
| photometric amplitude vs Ro, solar-mass Kepler stars | R_per ~ 1% plateau for 0.2 < Ro/Ro_sun < 0.4; then d ln R_per / d ln Ro = -2.5 to -4.5 | Masuda 2022, ApJ (arXiv:2206.01595) |
| Kepler R_per distribution (detected periods) | 5th / 50th / 95th percentile = 950 / 5600 / 22700 ppm | McQuillan, Mazeh & Aigrain 2014, ApJS (arXiv:1402.5694) |
| starspot lifetimes (Kepler, solar-type) | 10-350 d for spots of 0.1-2.3% of the hemisphere | Namekata et al. 2019, ApJ (arXiv:1811.10782) |

Two facts matter for the design:

1. **The shape is measured; the level is not settled.** Cao & Pinsonneault give a
   saturation knee at Ro/Ro_sun = 10^(-0.729-0.39) = 0.076 and a slope of -0.88 above it,
   but their Pleiades sample only reaches log Ro = -0.3 (Ro/Ro_sun ~ 0.2). The spectroscopic
   level at Ro_sun (2.9%, and 2% in M67 solar analogues) is ~10x the resolved sunspot
   coverage of the Sun at maximum (0.28%). Spectroscopic two-temperature fits also count
   spots that are too small, or too evenly spread, to modulate the light curve.
2. **Many spots is normal.** A spectroscopic coverage of 25-50% on saturated stars, with
   photometric amplitudes of only a few percent, requires many spots spread in longitude.
   A cap of 500 spots over 1000 days is not physical.

## Model

`create_data.SPOT_COVERAGE_DEFAULT`, column `Spot Coverage pct` (time-mean % of the whole surface):

    f_i = f_sun * max(Ro_i/Ro_sun, ro_sat)^-slope * 10^N(0, scatter_dex),   f_i <= f_max

| parameter | default | origin |
|---|---|---|
| f_sun | 0.003 | Sun at maximum (Shapiro 2014); calibration parameter, see below |
| slope | 0.881 | Cao & Pinsonneault 2022 eq. 5 |
| ro_sat | 0.076 | same, converted with their log Ro_sun = 0.39 |
| scatter_dex | 0.09 | same, 21% at saturation |
| f_max | 0.5 | TiO saturation level, Fang 2016 |

Ro uses the table's Cranmer & Saar 2011 tau_c with Ro_sun = 25.4 d / tau_c(5772 K). Only the
ratio Ro/Ro_sun is transferred between conventions; the mass dependence of the two
tau_c definitions differs, which is not corrected. The same law applies to subgiants
and to the active giants (no separate giant rule).

`Star.generate_spot_map` turns the coverage into a spot count:

    N = f * 41253 deg^2 * ndays / (<A> * L),   <A> = mean spot area, L = sqrt(pi/8) (tau_emerge + tau_decay)

with `<A>` from the size-bin distribution of `emerge_regions` (exact caps, radius
sqrt(2.5 bsize^2)) and L the time integral of the gaussian area evolution (radius factor
exp(-t^2/tau^2), so area ~ exp(-2 t^2/tau^2)). The count is Poisson around N; there is no floor,
so a quiet star can have very few spots. On the faculae smoke run (old rule), this formula
reproduces the simulator's own time-mean filling factor `mean_ff_sp` with median ratio 0.96
(5-95%: 0.58-1.32, from inclination, latitude and Poisson noise).

**Burn-in.** Spots also emerge during a burn-in before t = 0, lasting 2x the longer of the
spot and facular decay times, with the count scaled by (ndays + burn) / ndays. Coverage is
then stationary from the first day. Without it, coverage rose from zero over one spot
lifetime: a 60-day run reached ~2% instead of the 3.7-7.6% targets. With it, the same
stars give 3.4-8.1% (`mean_ff_sp`, visible disk). The legacy rule keeps its old behaviour
(first spot at t = 0).

**Run time.** The per-spot cost was a pure-Python visibility loop over every spot at every
time step, including long-decayed spots: gaussian-law radii never reach zero. The check
is now vectorised, and spots with radius < 1e-4 rad (~1e-8 of the disk) are skipped
(`rotator.MIN_SPOT_RAD`). On a 24 ppt light curve with 567 spots, the change in flux is
at most 0.03 ppm. Above `max_spots` spots per run (star.conf `[general] max_spots`,
default 10000, burn-in not counted), `Spot Max Area` is scaled up until the count fits:
same coverage, fewer and bigger spots. Large starspots are commonly interpreted as
unresolved groups of smaller spots. The per-star config records `n_spots`,
`n_spots_expected`, `spot_area_boost`, `spot_max_area_used` and `spot_burn_days`.
`--coverage_config '{"model": "legacy"}'` restores the old rule.

## Population preview (5000 stars, no light curves; counts per 1000-day window, burn-in excluded)

| f_sun | coverage % (5 / 50 / 95) | spots per 1000 d (5 / 50 / 95) | above 3000 / 10000 spots |
|---|---|---|---|
| old rule | 0.18 / 0.86 / 4.3 (smoke run) | 76 / 225 / 500 (24% at the cap) | - |
| 0.003 (resolved Sun) | 0.14 / 0.47 / 1.9 | 16 / 117 / 1006 | 1% / 0% |
| 0.01 | 0.48 / 1.55 / 6.3 | 54 / 390 / 3353 | 6% / 1% |
| 0.029 (spectroscopic) | 1.4 / 4.5 / 18 | 156 / 1132 / 9725 | 24% / 5% |

## Run time vs spot count

Single 1000-day Kepler light curve, 48000 steps at 30-min cadence, P = 1 d, one core:

| spots | before (Python visibility loop) | now |
|---|---|---|
| 0 | 12 s | 14 s |
| ~500 | 159 s | 71 s |
| ~1500 | 346 s | - |
| ~3000 | - | 108 s |
| ~10000 | - | 207 s |

## Calibration (to do)

Pick f_sun from light curves. The test: R_per percentiles against McQuillan 2014
(950 / 5600 / 22700 ppm), and R_per vs Ro/Ro_sun against Masuda 2022 (about a 1% plateau,
then a steep decline above Ro/Ro_sun ~ 0.4). The arms are f_sun = 0.003, 0.01 and 0.029,
about 200 stars each, with faculae on. McQuillan's percentiles are for stars with
*detected* periods, a sample biased toward large amplitudes, so the simulated
distribution should match the upper part and may fall below at the low end.
