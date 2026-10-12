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
| f_break | 0.06, provisional | own calibration (runs act_fb0300/0600/1200) |
| same law for dwarfs, subgiants, giants | - | own |
| tau_c conventions | only Ro/Ro_sun carried across papers | own |
| old single law | f_sun 0.005 | own calibration vs McQuillan 2014 |

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
| GIANT_LOGG 3.5, DYNAMO_TEFF_MAX 7000 K, subgiants always spotted | own |
| P_ACTIVE_GIANT 3% | own, rounded up from [paper] Ceillier 2017 (~2%) |
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

## Sun calibration (`scripts/sun_calibration.py`)

| item | status |
|---|---|
| i = 90 deg (Sun's ecliptic view is ~83 deg), no differential rotation, spots at 5-30 deg, one 1000-d window at fixed coverage | own |
