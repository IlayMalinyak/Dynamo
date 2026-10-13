# Assumptions and sources

Every modelling value on the `faculae` branch is tagged in the code:

- **[paper]**: value or functional form from a paper, cited in the comment.
- **[own assumption]**: our choice; not measured, not taken from a paper.
- **[own calibration]**: fitted by us, with this simulator, against published data.
- **[inherited]**: from the original code; source not re-checked.
- **[numerical]**: a numerical setting (tolerance, cutoff, quadrature).

The list below gathers the **own assumptions** and **own calibrations**, the places
where results depend on our choices. The code comment next to each value is the
reference.

## Spot coverage and spot properties (`create_data.py`)

| item | value | status |
|---|---|---|
| **activity scaling of spot properties: transition range** | **Ro/Ro_sun 0.2 (active) to 1.0 (solar)** | **own assumption, not a measurement.** Only the endpoints are measured. Range, log-linear shape and "same range for dT, size and Q" are our choice. The f_break calibration absorbs this choice. |
| solar spot dT | 600 K (contrast 0.33) | own pick inside [paper] Meunier 2010 (450-670 K, best 663 K) and Lagrange 2010 (550 K) |
| solar spot size | Spot Max Area x 1.2/12.2 | [paper] Baumann & Solanki 2005; own: <A> read as lognormal median, match on the area-weighted mean |
| solar Q | 10, network excluded; cap 40 | [paper] Meunier 2010 plage ~10x spots; own: network excluded, cap |
| coverage law junction | ro_flat = 0.2 | own (edge of the Cao & Pinsonneault data) |
| flat coverage 0.2-0.5 | - | [paper] Masuda 2022 R_per plateau; own calibration: flat R_per = flat coverage (R_per ~ f^0.58 in our simulator) |
| steep part 0.5 -> 1 | power law f_break -> f_tail | own functional form |
| f_tail | 0.1% at Ro_sun | own calibration (Sun anchor, docs/sun_calibration.md) |
| beyond Ro_sun | slope 0.881 | own (Cao's slope reused where no data exist) |
| coverage scatter | 0.09 dex at all Ro | [paper] Cao & P. at saturation; own: constant in Ro |
| R_per comparison excludes planet hosts | own (as McQuillan excluded KOIs) |
| f_break | 0.06 | own calibration (runs act_fb0300/0600/1200, docs/spot_coverage.md); depends on the 0.2-1 range above |
| same law for dwarfs, subgiants, giants | - | own |
| tau_c conventions | only Ro/Ro_sun carried across papers | own |
| old single law | f_sun 0.005 | own calibration vs McQuillan 2014 |

## Stellar priors (`create_data.draw_star_inputs`), checked 2026-10-13

Against Berger et al. 2020 (186k Kepler stars) crossed with Santos et al. 2019/2021 periods,
Teff <= 6500 K, logg >= 3.5. 5 / 50 / 95%:

| | ours | Berger, all Kepler | Santos periods x Berger | decision |
|---|---|---|---|---|
| mass [M_sun] | 0.62 / 0.96 / 1.29 | 0.68 / 0.99 / 1.35 | 0.60 / 0.90 / 1.33 | kept (inherited lognormal) |
| [Fe/H] | -0.28 / -0.02 / 0.26 | -0.29 / 0.00 / 0.21 | -0.25 / 0.00 / 0.25 | kept (inherited N(-0.03, 0.17)) |
| Teff [K] | 4024 / 5593 / 6367 | 4424 / 5692 / 6329 | 4082 / 5473 / 6317 | follows |
| age [Gyr], old prior truncnorm(2.5, 2) | 0.48 / 2.80 / 5.73 | 1.75 / 5.43 / 14.3 | 1.59 / 5.75 / 15.8 | **replaced** |
| P [d], old age prior | 3.5 / 17.3 / 32.6 | - | 3.7 / 18.9 / 38.8 | too fast |
| P [d], new age prior (below) | 4.4 / 25 / 45 | - | | slower than the detected sample, as expected: Kepler's rotation detection is nearly complete only to ~3 Gyr (Bouma 2024, citing Masuda 2022) |

**Age prior** (`sample_age`): p(t) ~ 1 + 0.75 min(t, 3 Gyr) on 0.05-10 Gyr.
- [paper] Bouma et al. 2024 (arXiv:2410.06246): rotation and lithium ages of 23,813 Kepler stars
  (3800-6200 K) give 2.1x as many 2-3 Gyr stars as 0-1 Gyr stars, and a star formation rate
  falling by 2.81 +- 0.12 over 3 Gyr (kinematic heating plus declining SFR). Slope 0.75/Gyr
  reproduces both. Medians of their quality-0 table give a steeper 2.5; we use their
  posterior-based numbers.
- [own assumption] linear shape; flat density from 3 to 10 Gyr. Bouma's calibration ends at
  4 Gyr and Kepler's period completeness drops beyond ~3 Gyr, so older ages are unconstrained.
- Berger et al. 2020 isochrone ages are not used (56% median uncertainty, tail beyond the age
  of the Galaxy).
- 5% of the new table is younger than 1 Gyr (old prior: 10%).

## Table columns not used by the default model

Kept so the optional paths keep working; do not read them as physical inputs of the default
(`model` "broken", `lat_model` "gaussian") simulation.

| column | status |
|---|---|
| `alpha/H` | drawn, never used (YREC interpolation takes mass, [Fe/H], age) |
| `Activity Rate` | used only by `model` "legacy" (spot count, spot-band amplitude); 0 for stars without a dynamo |
| `Spot Min`, `Spot Max` | used only by the "band" latitude model; with "gaussian" they are mean +- 2 width, for reference |
| `CDPP`, `Outlier Rate`, `Flicker Time Scale` | 0 for clean tables (`--add_noise` off) |

`Distance` is now derived from a drawn `Kp` (magnitude-limited survey); noise added at training
time should use `Kp` (or `background.add_background(..., mag=None)`, which recovers it from
`Distance`).

## Population priors (`create_data.py`)

| item | status |
|---|---|
| Spot Max Area 3-50 deg^2, log-uniform | own |
| spot contrast scatter N(1, 0.2) on [0.5, 1.5] | own |
| Q prior median 3 | [paper, not verified] Chapman 1997 (not on arXiv); own: 0.4 dex scatter, bounds 0.3-20, the -0.3 activity exponent |
| facular contrast scale N(1, 0.3) on [0.3, 2] | own |
| facular lifetime ratio log-uniform 1.5-5 (star default 3) | own |
| spot lifetime truncnorm(6, 2, 1.5, 15) rotations | own ("arbitrary" in the original code) |
| cycle length truncnorm(10, 2.8, 2, 25) yr | inherited ("Reinhold", not verified) |
| no giants in tables (KEEP_GIANTS = False, redrawn like hot stars) | own: grid has no red clump, giant periods were placeholders, giant spots uncalibrated, no seismic inclination signal (2026-10-13) |
| GIANT_LOGG 3.5 (giant definition), subgiants always spotted | own |
| DYNAMO_TEFF_MAX 6500 K (was 7000 K) | own: edge of the calibration data (McQuillan 2014 < 6500 K) |
| TEFF_SAMPLE_MAX 6500 K: hotter dwarfs/subgiants redrawn when the table is built | own (same population as cutting, no lost rows) |
| P_ACTIVE_GIANT 3% (only with KEEP_GIANTS) | own, rounded up from [paper] Ceillier 2017 (~2%) |
| active-giant activity 1-30 | own |
| latitude population: all numbers and the form (mean linear in log Ro, linear drift) | own; [paper] only the solar centroid range (Hathaway 2015) and the sign of the Ro trend |

## Spots, faculae, light curves (`spots.py`, `star.py`, `rotator.py`, `contrasts.py`, `nbspectra.py`, `spectra.py`)

| item | status |
|---|---|
| spot size bins (5 bins, d lnA 0.5, p_k ~ e^k) | inherited (butterpy / Aigrain 2015 style) |
| tau_emerge = min(2, P decay/10), tau_decay = max(P decay, 50) d | inherited |
| burn-in = 2 x longest decay time | own |
| spot count Poisson around the expected number | own |
| max_spots 10^4, bigger spots above it | own (run-time) |
| MIN_SPOT_RAD 1e-4 rad | numerical (<= 0.03 ppm) |
| facular radius evolution (gaussian, own tau_fac) | own functional form |
| per-spot Q scatter 0.3 in ln | own |
| spot dT band +-10% | own |
| facular intensity: linear blend toward a T + 300 K model | own approximation; [paper] the dT(mu) law is Meunier 2010 (coefficient corrected 407.4 -> 407.7 on 2026-10-12) |
| facular contrast `scale` as the unknown Teff dependence | own |
| overlaps between different features not resolved | own simplification |
| spectra: faculae radiate the disk-averaged facular intensity | own approximation |
| hemispheres 50/50 | own |
| gaussian latitudes with linear drift | own (the Sun's drift is not linear) |

## Background, noise, magnitudes (`background.py`)

| item | status |
|---|---|
| granulation/oscillation taper at 7000 K, width 150 K | own |
| Kepler/TESS read noise and floor (6000 e-, 10 ppm; 4000 e-, 85 ppm) | own settings, not fitted to a CDPP relation |
| magnitude prior p(m) ~ 10^(0.3 m), ranges Kp 9-16, T 7-13 | own |
| Kp / TESS colour offsets | own approximation; no extinction |
| mode linewidth 0.95 (Teff/Tsun)^10, clip 0.05-10 muHz | own normalisation; Teff trend cited, not verified |
| d01 = 0.025 Dnu, d02 = 0.12 Dnu constant | own simplification (Mosser 2011 pattern, not verified) |
| +-4 sigma mode orders, 1e-3 A_max cut | numerical |

## Planets (`planet_params.py`)

| item | status |
|---|---|
| A_MIN_RSTAR = 3 | own |
| obliquity: aligned spread 10 deg, misaligned fraction 0.1 / 0.5, isotropic misaligned, step at the Kraft break | own numbers and forms; [paper] Kraft break 6250 K and the trend (Winn 2010; Albrecht 2012, 2022) |
| uniform spin azimuth; coplanar multi-planet systems | own |
| RV K without (1 - e^2)^-1/2 | inherited |
| planet radius from mass | [paper] Baron, Mueller, Helled et al. 2024 Eq. 1 (fixed 2026-10-13: exponents were inverted, median planet was 0.24 R*) |
| planet mass prior | [paper, not verified] "Malhotra 2015" (inherited) |

## Sun calibration (`scripts/sun_calibration.py`)

| item | status |
|---|---|
| i = 90 deg (Sun's ecliptic view is ~83 deg), no differential rotation, spots at 5-30 deg, one 1000-d window at fixed coverage | own |
