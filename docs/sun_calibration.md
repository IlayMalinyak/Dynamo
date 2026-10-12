# Sun calibration

Question: does the simulator, given the Sun's own spot parameters and coverage, give the
Sun's measured rotational variability? And are the population's spot parameters solar at
Ro_sun? Runs: `scripts/sun_calibration.py` (solar twin: Teff 5772 K, R = 1, P = 25.4 d,
equator-on, Kepler band, 30-min cadence, latitudes 5-30 deg). All runs: `docs/sun_calibration.csv`.
R_per is McQuillan's statistic: median over rotation-period windows of the 5-95% range.

## Solar numbers (checked against the arXiv sources)

| quantity | value | source |
|---|---|---|
| spot / facular disc coverage, 1991 maximum | 0.28% / ~3% | Shapiro et al. 2014, arXiv:1406.2383, Sect. 3 |
| spot / facular disc coverage, 1996 minimum | 0.003% / 0.36% | same |
| fits vs Ca II S | A_S = 0.105 - 1.315 S + 4.102 S^2, A_F = -0.233 + 1.400 S | same, Eqs. 1-2 |
| plage area vs spot area (MDI, cycle 23) | plage ~ 10x spots | Meunier et al. 2010, arXiv:1001.1638, Sect. 2.1 |
| spot temperature contrast fitted to TSI | single-temperature spots: 663 K (450-670 K depending on area scaling) | same, Sect. 2.2.2 |
| sunspot group areas (RGO 1874-1976) | lognormal, <A> = 62.2 MSH, sigma_A = 2.45 (total area, maximum development) | Baumann & Solanki 2005, astro-ph/0510516, Table 1 |
| spot lifetimes | A_0 = D_GW T, D_GW = 10.89 +- 0.18 MSH/day | Petrovay & van Driel-Gesztelyi 1997, astro-ph/9706029 |
| sunspot area vs sunspot number | area [MSH] ~ 16.7 SSN | Hathaway 2015, arXiv:1502.07020, Sect. 3.2 |
| Sun's R_var, 30-day windows (VIRGO TSI, cycle 23) | 20th / 50th / 95th percentile / maximum = 0.24 / 0.65 / 1.7 / 2.5 ppt | Basri et al. 2013, arXiv:1304.0136, Tables 1-2 |
| Sun's R_var, 90-day windows, median over 4 yr | median 0.7 ppt, maximum <~ 2 ppt (140 yr reconstruction) | Reinhold et al. 2020, arXiv:2005.01401 |
| Kepler band vs TSI | Kepler R_var ~ 10% larger | same, Supplement |
| what drives it | spot-dominated from the ecliptic in the Kepler band | Shapiro et al. 2016, arXiv:1602.04447 |

1 deg^2 of the sphere = 48.5 MSH; full-surface fraction = MSH x 1e-6 / 2. Not found in the
literature: a cycle-averaged spot coverage. From Hathaway's 16.7 x SSN it is roughly
0.04-0.05% of the surface (my estimate, not a published value).

## Solar parameters in simulator units

| parameter | population default at Teff 5772 K | solar |
|---|---|---|
| spot dT | 1822 K (Berdyugina 2005 relation, scale ~ N(1, 0.2)) | ~600 K, `Spot Contrast Scale` 0.33 |
| `Spot Max Area` | 3-50 deg^2, log-uniform (area-weighted mean spot 530-8850 MSH) | ~1.2 deg^2 (area-weighted mean 207 MSH, Baumann & Solanki) |
| `Facula Q` | median 3 | ~10 (plage only) |
| spot lifetime | `Decay Time` ~ 6 rotations (tau_d = 152 d) | most groups live days to weeks (`Decay Time` 1.5 -> tau_d floor of 50 d) |

## Results

One factor at a time, from the population defaults (coverage 0.3%, 400 d, 2 seeds):

| change | R_per [ppt] |
|---|---|
| defaults (dT 1822 K, max area 10, Q 3, decay 6) | 5.2 |
| dT 547 K / 911 K | 2.1 / 3.2 |
| max area 1 / 3 deg^2 | 2.1 / 3.2 |
| Q 0 / 8 | 6.1 / 5.2 |
| decay 1.5 / 0.5 | 4.3 / 5.5 |

The population's spots at Ro_sun are each ~2.5x too dark and ~10x too large for the Sun.

Solar parameters (dT 600 K, max area 1.2 deg^2, decay 1.5), 1000 d, 3 seeds, R_per [ppt]:

| spot coverage | Q = 10 (plage) | Q = 0 | Q from Shapiro A_F/A_S (incl. network) | Sun |
|---|---|---|---|---|
| 0.003% (minimum) | 0.07 | 0.02 | 1.5-1.7 (Q = 185) | 20th percentile 0.24 |
| 0.03% | 0.40 | 0.25 | 1.1-1.8 (Q = 32) | |
| 0.1% | 0.63 | 0.49 | 1.2-1.4 (Q = 17) | median 0.65-0.7 |
| 0.28% (maximum) | 1.25 | 0.89 | 1.2-1.3 (Q = 11) | 95th percentile 1.7, maximum 2-2.5 |

1. **With solar inputs, the simulator gives the Sun.** Plage Q = 10 reproduces the median
   (0.63 vs 0.65-0.7 ppt) and is 1.4-2x low at maximum (1.25 vs 1.7-2.5 ppt). The low
   maximum likely comes from the missing large-group tail: the size bins reach only
   ~460 MSH at max area 1.2 deg^2, while the largest solar groups exceed 3000 MSH.
2. **Faculae must be plage only.** Shapiro's facular area includes the network, which is
   spread over the disc. Put into rings around a few spots at minimum (Q = 185), it gives
   1.5 ppt instead of < 0.24.
3. **The population is not solar at Ro_sun.** The broken coverage law (0.04% at Ro_sun)
   with population spots gives ~0.9 ppt for Sun-like stars, close to the Sun's 0.65-0.7.
   It gets there with ~10x larger, ~3x darker spots on a smaller coverage. Coverage at
   Ro_sun is near the estimated solar cycle mean (0.04-0.05%), so the law's level is
   plausible, but the per-spot properties are not solar.

## Population change: spot properties scale with activity

Doppler-imaging and TiO spot temperatures (the Berdyugina 2005 relation) and large spots
describe active stars; the Sun's spots are lighter and smaller. Each star's drawn
`Spot Contrast Scale`, `Spot Max Area` and `Facula Q` are now moved toward solar values
with Rossby number (`create_data.SPOT_PROPS_DEFAULT`, `scale_spot_properties`):

| x = Ro/Ro_sun | weight w | contrast scale | max area | Q |
|---|---|---|---|---|
| <= 0.2 | 0 (active) | drawn, median 1 (dT 1822 K at 5772 K) | drawn, 3-50 deg^2 | drawn, median 3 |
| 0.5 | 0.57 | x 0.53 | x 0.27 | x 2.0 |
| >= 1 | 1 (solar) | x 0.33 (dT ~ 600 K) | x 0.098 (median 1.2 deg^2) | x 3.3 (median 10, cap 40) |

with factor^w and w linear in log x between 0.2 and 1. The per-star scatter of each
draw is kept.

> **ASSUMPTION, NOT A MEASUREMENT.** Only the two endpoints are measured: the active
> values (Berdyugina 2005 spot temperatures, the drawn spot sizes) and the solar values
> (table above). **Where and how** spot temperature, spot size and Q change between them
> is not measured and not taken from any paper:
>
> - **ro_active = 0.2 and ro_solar = 1.0 (Ro/Ro_sun) are our choice.** 0.2 reuses the edge
>   of the Cao & Pinsonneault data; 1.0 is the Sun. No study we found measures spot
>   temperature, spot size or Q against Ro over 0.2-1.
> - **The shape (geometric, linear in log Ro) is our choice.**
> - **All three properties are assumed to change over the same range.**
>
> The f_break calibration absorbs this choice: a different range would give a different
> f_break. Sensitivity can be tested with
> `--coverage_config '{"props_config": {"ro_active": ..., "ro_solar": ...}}'`. Spot lifetimes are not scaled (`Decay Time` effect is small: 5.2 -> 4.3 ppt
for 6 -> 1.5 rotations). `--coverage_config '{"props": "fixed"}'` keeps the drawn values;
model "legacy" also keeps them.

The coverage law is anchored to the Sun: f(Ro_sun) = `f_tail` = 0.1%, the coverage at
which solar-property spots give the Sun's median R_per. Between Ro/Ro_sun = 0.5 and 1 it
is a power law from `f_break` to `f_tail`. `f_break` (plateau level) is the remaining
calibration parameter.

**Calibration (done, 2026-10-12):** runs `dataset_cov_act_fb0300`, `_fb0600`, `_fb1200`
(f_break 0.03 / 0.06 / 0.12). **f_break = 0.06**: Masuda plateau 10.7 ppt (target 10.8), slope
-4.2 (target -4.6 +- 0.9), Sun-like stars 0.73 ppt (Sun 0.65-0.7); planet hosts excluded. The rough estimate above
gave ~0.07. Details and the mismatches (McQuillan 95th percentile high): docs/spot_coverage.md.
