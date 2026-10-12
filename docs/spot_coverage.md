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
| photometric amplitude vs Ro, 4000-6250 K Kepler dwarfs (Cranmer & Saar tau_c, Ro_sun ~ 2) | all stars: R_per = 10.8 +- 0.4 ppt at Ro = 0.84 +- 0.02 (Ro/Ro_sun 0.43), slope -0.04 +- 0.09 below and -2.6 +- 0.1 above; Kp < 12 (least detection bias): break at Ro = 0.99 (0.50 Ro_sun), slope -4.6 +- 0.9 above; scatter 0.24 dex | Masuda 2022, ApJ (arXiv:2206.01595), Sec. 3.2-3.3 |
| Sun vs Kepler solar-like stars with detected periods | median variability of the stars 5x the Sun's | Reinhold et al. 2020, Science 368, 518, doi:10.1126/science.aay3821 |
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

`create_data.SPOT_COVERAGE_DEFAULT`, column `Spot Coverage pct` (time-mean % of the whole surface),
`f_i = law(Ro_i/Ro_sun) * 10^N(0, scatter_dex)`, `f_i <= f_max`, with the broken law
(`create_data.spot_coverage_law`, model "broken"), x = Ro/Ro_sun:

| range | coverage | origin |
|---|---|---|
| x < ro_sat = 0.076 | saturated, 2.34 f_break | Cao & Pinsonneault 2022 eq. 5 knee |
| 0.076 < x < ro_flat = 0.2 | f_break (x/0.2)^-0.881 | Cao & Pinsonneault slope (their data reach x ~ 0.2) |
| 0.2 <= x <= ro_break = 0.5 | f_break = 0.02 | Masuda: R_per flat (slope -0.04 +- 0.09); level calibrated below |
| 0.5 < x <= ro_tail = 0.8 | f_break (x/0.5)^-7.93 | Masuda Kp < 12: R_per slope -4.6, and R_per ~ f^0.58 per star |
| x > 0.8 | continues with slope -0.881 | end of Masuda's steep range; the Santos sample flattens at x >~ 1 |

scatter_dex = 0.09 (Cao & Pinsonneault, 21% at saturation), f_max = 0.5 (TiO saturation,
Fang 2016). Resulting medians: 4.7% saturated, 2% on the plateau, 0.14% at x = 0.7,
0.04% at Ro_sun. The first calibration's single law, f_sun * max(x, ro_sat)^-0.881 with
f_sun = 0.005, remains as model "rossby".

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
| **0.005 (default)** | 0.23 / 0.78 / 3.2 (0.003 row x 5/3) | 27 / 195 / 1677 (same scaling) | - |
| 0.01 | 0.48 / 1.55 / 6.3 | 54 / 390 / 3353 | 6% / 1% |
| 0.029 (spectroscopic) | 1.4 / 4.5 / 18 | 156 / 1132 / 9725 | 24% / 5% |
| **broken law (default)** | 0.019 / 0.41 / 3.0 | 3 / 84 / 2103 | 2.5% / 0%; 1% below one spot |

## Run time vs spot count

Single 1000-day Kepler light curve, 48000 steps at 30-min cadence, P = 1 d, one core:

| spots | before (Python visibility loop) | now |
|---|---|---|
| 0 | 12 s | 14 s |
| ~500 | 159 s | 71 s |
| ~1500 | 346 s | - |
| ~3000 | - | 108 s |
| ~10000 | - | 207 s |

## Calibration

Data: `docs/spot_coverage_calibration.{md,csv,png}`, from `scripts/coverage_calibration_report.py`.
One 200-star table (172 with a dynamo), written four times by `scripts/make_coverage_tables.py`
with only the coverage column changed. 1000-day Kepler light curves, faculae on, no noise.
Athena jobs 178155-178158, 178482-178483. R_per for main-sequence stars below 6500 K (150 per arm):

| arm | R_per 5 / 50 / 95% [ppt] | bimodality of log R_per | spots median / max | rho(R_per, Ro) | rho(R_per, Spot Max Area) |
|---|---|---|---|---|---|
| old rule (500 cap) | 1.03 / 6.59 / 23.8 | 0.42 | 270 / 500 | -0.25 | +0.75 |
| f_sun = 0.003 | 1.04 / 4.66 / 16.3 | 0.30 | 210 / 5591 | -0.59 | +0.40 |
| f_sun = 0.01 | 2.16 / 9.94 / 28.9 | 0.37 | 674 / 11600 | -0.56 | +0.49 |
| f_sun = 0.029 | 4.38 / 17.8 / 48.5 | 0.39 | 1946 / 14860 | -0.53 | +0.50 |
| McQuillan 2014 | 0.95 / 5.6 / 22.7 | | | | |

Against Masuda 2022 (4000-6250 K dwarfs; `plateau` and `slope` columns of the readout):
plateau R_per at 0.2 < Ro/Ro_sun < 0.5 is 9.4 / 5.4 / 11.5 / 19.8 ppt (old rule, 0.003,
0.01, 0.029; Masuda 10.8) and the slope over 0.5-0.8 is +0.3 / -0.7 / -0.5 / -0.3
(Masuda -4.6 +- 0.9).

Findings:

1. **The second amplitude peak is gone.** Sarle's coefficient drops from 0.42 to 0.30
   (0.555 would suggest two modes). Amplitude now follows the Rossby number (rho -0.25 ->
   -0.59) instead of mostly the spot-size draw (+0.75 -> +0.40).
2. **f_sun = 0.029 (spectroscopic level) is ruled out:** median 3x Kepler.
3. **McQuillan sets f_sun ~ 0.004-0.006.** R_per scales as f_sun^0.63 at the median and
   f_sun^0.48 at the 95th percentile (from the 0.003 and 0.01 runs). Matching the median
   needs 0.004 and the 95th percentile 0.006. McQuillan counts only stars with detected
   periods, which favours large amplitudes. **Default: f_sun = 0.005.**
4. **The shape did not match Masuda 2022** (fixed below): the single -0.88 law gives an
   R_per slope of about -0.5, against -4.6.

### Slope fix: broken law

In these runs, amplitude per unit coverage hardly depends on Ro (rho(R_per / f^0.58, Ro) =
-0.11; spot size matters more, +0.41), and per star R_per ~ f^0.58 (median of the paired
arms; 0.52-0.62 in every Ro bin, close to the sqrt(N) of spots at random longitudes). So
Masuda's R_per law maps directly onto coverage: flat for 0.2-0.5 Ro_sun, slope
-4.6/0.58 = -7.93 above. Masuda uses the same Cranmer & Saar tau_c as the table
(Ro_sun ~ 2; ours 1.97), so no conversion is needed. The bright-star (Kp < 12) slope is
used because the simulation has no detection bias; the all-star -2.6 is shallower
because weak modulation is missed (Masuda Sec. 3.3).

f_break was set by predicting each star's R_per under candidate laws, interpolating its
log R_per vs log f between the three paired arms (slope 0.58 beyond them):

| law | plateau [ppt] | slope 0.5-0.8 | R_per 5 / 50 / 95% | f(Ro_sun) |
|---|---|---|---|---|
| f_break 0.02, steep to 1.0 | 10.3 | -4.9 | 0.18 / 5.78 / 22.6 | 0.008% (6% of stars < 0.5 spots) |
| **f_break 0.02, steep to 0.8 (default)** | **10.3** | **-4.9** | **0.37 / 5.78 / 22.6** | **0.04%** |
| target | 10.8 (Masuda) | -4.6 +- 0.9 | 0.95 / 5.6 / 22.7 (McQuillan) | |

One law now fits Masuda's plateau and slope and McQuillan's median and 95th percentile:
the earlier conflict between levels (0.005 for McQuillan, 0.01 for Masuda's plateau) came
from the slope. The 5th percentile sits below McQuillan, as expected for a catalogue of
detected periods. Ending the steep part at 0.8 rather than 1.0 changes nothing over
Masuda's range; it keeps Sun-like stars spotted.

Open: at Ro_sun the law gives 0.04% coverage, below the Sun's resolved sunspot coverage
(0.28% at maximum). Kepler stars near Ro_sun are themselves more variable than the Sun
(Reinhold 2020), so the Sun is not used as an anchor. The low value may mean the
simulation over-produces amplitude per unit spot area at low activity (e.g. too-large
spots or too little facular cancellation), which this calibration absorbs into f.

**Superseded by the Sun calibration** (docs/sun_calibration.md): the population's spots at
Ro_sun were ~3x darker and ~10x larger than the Sun's. Spot properties now scale with
activity (**the 0.2-1 Ro/Ro_sun transition range is an assumption, not a measurement**;
see docs/sun_calibration.md), the law is anchored at f(Ro_sun) = 0.1%, and f_break was recalibrated (below). The
`dataset_cov_broken` confirmation run was not made.

Caveats: no photometric noise, no period-detection selection, 150 stars per arm.

**Memory.** The first f_sun = 0.01 / 0.029 runs hung at the 16 GB job limit. The rotator kept
every spot's position at every step (n_steps x n_spots x 3), only for the `plot_map`
figure: ~17 GB at 48000 steps and ~14700 spots including burn-in. A killed worker hangs the
multiprocessing pool rather than failing the job. The positions are now kept only with
`plot_map`. Per-task peak memory fell from 10-16.6 GB to 1.8-2.7 GB.

## Calibration of f_break with activity-scaled spots (2026-10-12)

Runs `dataset_cov_act_fb0300/0600/1200`: the default model (Sun-anchored broken law,
activity-scaled spot properties) with f_break = 0.03 / 0.06 / 0.12, same 200 stars as the
arms above. `python scripts/coverage_calibration_report.py dataset_cov_legacy
dataset_cov_act_fb0300 dataset_cov_act_fb0600 dataset_cov_act_fb1200 --out
docs/spot_coverage_calibration_fbreak` (all seven arms: `docs/spot_coverage_calibration_act`).
Light curves: `docs/lc_gallery_fb0600.png` (`scripts/lc_gallery.py`).

| arm | plateau 0.2-0.5 [ppt] | slope 0.5-0.8 | R_per 0.8-1.0 Ro_sun, 0.9-1.1 M_sun [ppt] | R_per 5 / 50 / 95% [ppt] | stars above max_spots |
|---|---|---|---|---|---|
| target | 10.8 (Masuda 2022) | -4.6 +- 0.9 (Masuda, Kp < 12) | 0.65-0.7 (Sun) | 0.95 / 5.6 / 22.7 (McQuillan 2014) | |
| f_break 0.03 | 7.1 | -3.1 | 0.57 | 0.24 / 3.8 / 19.9 | 3% |
| **f_break 0.06 (default)** | **10.7** | **-4.2** | **0.73** | **0.24 / 6.4 / 27.5** | **9%** |
| f_break 0.12 | 15.1 | -5.0 | 0.63 | 0.24 / 9.2 / 39.6 | 17% |

Planet hosts excluded (12% of stars), as McQuillan excluded KOIs: transits of planets up
to 0.3 R* enter the 5-95% range (one host reached 47 ppt on 0.04% spot coverage). With
hosts included the 0.06 arm gave plateau 10.8, slope -3.9, percentiles 0.25 / 7.0 / 33.4.

**Chosen: f_break = 0.06** [own calibration]. It matches the Masuda plateau and slope, and
Sun-like stars sit at the Sun's median.

Not matched:
- McQuillan's 95th percentile (27.5 vs 22.7 ppt; the fast active rotators).
- McQuillan's 5th percentile (0.24 vs 0.95 ppt). This comparison is biased: McQuillan lists
  only stars with a detected period, and quiet stars are missing from it.
- The steep drop above Ro/Ro_sun 0.5 comes partly from the [own assumption] 0.2-1 range of
  the spot-property scaling, not from f_break alone.
- 8% of stars exceed max_spots and get larger spots instead of more spots.
- ~200 stars per arm; the Ro-bin medians rest on few stars.

Flat light curves (24 of 200 in the 0.06 arm) are all stars without a dynamo (inactive
giants, Teff > 7000 K). Every spotted star varies.

**Hot stars.** Above ~6250 K (the Kraft break) the light curves turn from dip-dominated to
brightening-dominated (median skewness -0.3 below 5000 K, +0.2 at 6250-6500 K, +0.3 at
6500-7000 K). At 6500-7000 K the Cranmer & Saar tau_c is short, so Ro/Ro_sun reaches 2-9 even
at P < 1 d; coverage falls to ~0.02-0.06%, Q is at its cap, and the light curve is a few
isolated facular brightenings fading over ~100 d (`docs/lc_gallery_fb0600.png`, bottom). No
calibration data exist there (McQuillan < 6500 K, Masuda 4000-6250 K). **DYNAMO_TEFF_MAX
lowered from 7000 to 6500 K** [own assumption]: stars above it get no spots or faculae. The
6250-6500 K stars stay (inside McQuillan's sample) but are facula-leaning and only weakly
constrained.
