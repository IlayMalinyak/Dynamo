# Dynamo: faculae + dataset hygiene — implementation plan

Audience: the model/engineer implementing this. Work in `/home/ilay.kamai/work/Dynamo`
on a new branch `faculae`. One commit per Part (A, B, C, D, E). Run the tests of
each Part before moving on. Do **not** touch the grid geometry
(`nbspectra.generate_grid_coordinates_nb`), the planet code, or the PHOENIX
download/interpolation code except where stated.

Conventions used in the code and kept here:
* `i` = inclination in **degrees**, 0 = pole-on, 90 = equator-on (`Star.inclination`).
* Spot map row = `[t_emerge, duration, colatitude_deg, longitude_deg, radius_deg, ...]`
  (`Star.generate_spot_map`, `Rotator.compute_spot_position_vec`).
* Per-ring brightness arrays (`brigh_grid_ph/sp/fc`) are indexed by ring; ring `k`
  has `mu = cos_centers[k]`. The flux of a time step is
  `flx_ph - sum_pixels[(1-aph)*brigh_ph] + asp*brigh_sp + afc*brigh_fc`
  (`nbspectra.loop_generate_rotating_lc_nb`, line ~300).
* `Q = facular_area_ratio` = facular area / spot area (Starsim convention): the
  facula is an annulus around the spot with outer radius `r_spot*sqrt(1+Q)`.

---------------------------------------------------------------------------

## Part 0 — pre-existing bug found during implementation (FIXED, commit 4035b2e)
The PHOENIX SpecInt loaders reversed the mu labels but not the intensity rows,
so every mu was paired with the intensity from the other side of the disk; the
light-curve integrator compensated only for the central element. A 5 deg spot
gave an 18-78% dip at disk centre (grid dependent) and almost no signal
elsewhere. All light curves (and spectra) generated before this fix are wrong.
The T-B0 reference is therefore master + this fix, not plain master.

## Part A — hygiene (must be done first; no new physics)

### A1. Per-star random seed
`Dynamo/star.py:392` passes `seed=1234` to `emerge_regions` for **every** star.
* Add attribute `self.seed = None` in `Star.__init__`; set it in
  `set_stellar_parameters` from `params_dict.get('Seed')`.
* `generate_spot_map`: `seed=self.seed` (None → unseeded).
* `create_data.generate_simdata`: add column `sims['Seed'] = np.arange(Nlc) + base_seed`
  (`base_seed` new CLI arg, default 0). `simulate_one` also calls
  `np.random.seed(int(sim_row['Seed']))` at the top so spectra-epoch choice and
  noise are reproducible per star.

### A2. Truncate unbounded normals (`create_data.py`)
* line 152 `clen`: replace with `truncated_normal_dist(10, 2.8, 2, 25, Nlc)`.
* line 156 `tau_evol`: replace with `truncated_normal_dist(6, 2, 1.5, 15, Nlc)`.
* `convective_shift`: leave.
* Add an assertion block after the table is built: all of
  `Cycle Length > 0`, `Decay Time > 0`, `Spot Max > Spot Min`, `Period > 0`.

### A3. Stars that should not have a spot dynamo
In `generate_simdata`, after `teff` is known:
* `no_dynamo = (teff > 7000) | (~is_ms)`  → set `Activity Rate` to `0.02`
  (so `emerge_regions` yields the 20-spot floor; then override) **and** add
  column `sims['Has Dynamo'] = ~no_dynamo`.
* `Star.generate_spot_map`: if `getattr(self, 'has_dynamo', True)` is False,
  produce an **empty** spot map (`np.zeros((0, 16))`) and skip emergence.
  Check that `compute_forward` handles an empty map (rotator already branches
  on `spot_map.size == 0`; verify it returns a flat light curve). Set
  `self.has_dynamo` in `set_stellar_parameters` from `params_dict.get('Has Dynamo', True)`.

### A4. Teff-dependent spot contrast
Replace `star.py:289-296` (contrast forced to 200–250 K) with a function in a
new module `Dynamo/contrasts.py`:

```python
def spot_contrast_berdyugina(teff):
    """Berdyugina 2005 (Living Rev. Sol. Phys. 2, 8) polynomial fit,
    spot-minus-photosphere temperature difference in K (positive = cooler)."""
    return 3.58e-5 * teff**2 + 0.249 * teff - 808.0
```
(≈1840 K at 5800 K, ≈500 K at 3500 K.) In `set_stellar_parameters`:
`dT = spot_contrast_berdyugina(Teff) * params_dict.get('Spot Contrast Scale', 1.0)`,
`spot_T_contrast_min = 0.9*dT`, `spot_T_contrast_max = 1.1*dT`, `spot_T_contrast = dT`.
In `generate_simdata` add column `Spot Contrast Scale ~ truncnorm(1.0, 0.2, 0.5, 1.5)`.
Clamp `temperature_spot >= 2300` (PHOENIX grid floor).

### A5. Spot areas (compensates for A4)
With realistic contrasts the current `max_area = 50 deg²` gives amplitudes far
above Kepler. Make `max_area` a per-star parameter: column
`sims['Spot Max Area'] = 10**np.random.uniform(np.log10(3), np.log10(50), Nlc)`
(deg²), read in `set_stellar_parameters` into `self.spot_max_area`.
Target after A4+A5 (check on 200 stars, Kepler band, 1000 d, no noise):
median of `(q95 - q05)/median` of the flux between 3 and 8 ppt, 5th percentile
below 1 ppt. If not, widen the lower bound of `Spot Max Area` (to 1 deg²), do
not touch contrasts.

### Tests for Part A (`tests/test_hygiene.py`)
* T-A1: two `Star` objects with identical parameters and different `Seed`
  produce different spot maps; same seed → identical maps.
* T-A2: `generate_simdata(N=2000)` table passes the assertions in A2.
* T-A3: a star with `Has Dynamo=False` returns a light curve with
  `max|flux/median - 1| < 1e-12`.
* T-A4: `spot_contrast_berdyugina(5800)` within 50 K of 1840.

---------------------------------------------------------------------------

## Part B — facular brightness with centre-to-limb variation (photometry)

### B1. Facular temperature contrast vs μ
Add to `Dynamo/contrasts.py`:
```python
def facula_dT_mu(mu, scale=1.0):
    """Facular temperature excess (K, positive = hotter) vs mu.
    Solar fit of Meunier et al. 2010 (A&A 512, A39), as used in Starsim and
    SOAP 2.0: dT = 250.9 - 407.4 mu + 190.9 mu**2 (34 K at disk centre,
    251 K at the limb). `scale` is a star-level multiplier (unknown Teff
    dependence; default 1)."""
    return scale * (250.9 - 407.4*mu + 190.9*mu**2)
```
New config keys in `star.conf` `[star]`: `facula_contrast_scale : 1.0`,
`facula_lifetime_ratio : 3.0`. Read them in `_initialize_star_params`.
`set_stellar_parameters` reads `params_dict.get('Facula Q', 0.0)` →
`self.facular_area_ratio`, `'Facula Contrast Scale'` → `self.facula_contrast_scale`,
`'Facula Lifetime Ratio'` → `self.facula_lifetime_ratio`.
Remove the now-unused `facula_T_contrast` and `temperature_facula` property
(keep reading the conf key for backward compatibility; ignore it).

### B2. Per-ring facular brightness grid
`star.py` `compute_forward`, lines ~524-555. Currently the facular grid is
built from `photo_flux` (so faculae do nothing). Replace with:

1. `DT_REF = 300.0` K. Interpolate PHOENIX once more:
   `_, _, hot_flux = spectra.interpolate_Phoenix_mu_lc_with_metallicity(self, T_ph + DT_REF, logg, feh)`.
2. For each photometric filter, compute `brigh_grid_hot, _ = compute_immaculate_lc_with_vsini(... hot_flux ...)`
   exactly like `brigh_grid_ph`.
3. Per ring `k`: `w_k = facula_dT_mu(cos_centers[k], self.facula_contrast_scale) / DT_REF`
   and `brigh_grid_fc[k] = brigh_grid_ph[k] + w_k * (brigh_grid_hot[k] - brigh_grid_ph[k])`.
   (Filter integration is linear in intensity, so blending the ring
   brightnesses equals blending the intensities. Linear-in-T is accurate to
   a few percent for dT ≤ 300 K; acceptable.)
4. Keep `hot_flux` on `self` (`self._hot_flux`) for Part D.

If `self.facular_area_ratio == 0` skip steps 1–3 and set `brigh_grid_fc = brigh_grid_ph`
(fast path, identical to current behaviour).

### B3. Facular geometry and lifetime per spot
Spot map gains two columns (indices 14 and 15; the array is widened from 14
to 16 columns). IMPLEMENTATION NOTE: columns 5/6 were the original proposal but
the 'linear' evolution law reads column 5, and columns 4-13 are reserved for
radius coefficients, so the facular columns were moved to 14/15:
* col 14 = `q_i` facular area ratio of this spot (per-spot scatter:
  `q_i = Q * lognormal(0, 0.3)`),
* col 15 = `tau_fac_i = tau_decay * facula_lifetime_ratio`.
A map with fewer columns, or zeros there, falls back to the star-level Q and
`tau_decay * facula_lifetime_ratio`.

`Rotator.compute_spot_position_vec` (rotator.py ~331-352), gaussian branch:
* spot radius as now: `rad = R0 * factor_spot(t)`.
* facular radius: compute its own Gaussian factor with `tau_emerge` (same) and
  `tau_fac_i` for decay, then
  `rad_fac = deg2rad(R0) * sqrt(1 + q_i) * factor_fac(t)`.
  Note: facula outer radius uses the **undecayed** `R0` scaled by its own
  decay factor, so faculae persist after the spot has decayed. Store in
  `pos[:, 3]` as now. For the non-gaussian laws keep the old
  `rad_fac = deg2rad(rad)*sqrt(1+Q)`.
* `generate_rotating_photosphere_lc` visibility test (rotator.py:125) currently
  uses `spot_pos[i,2]*sqrt(1+Q)`; change to `max(spot_pos[i,2], spot_pos[i,3])`.
* `nbspectra.process_central_grid` / `process_outer_grid` (lines 333, 397)
  skip a spot when `spot_pos[idx][2] == 0`. Change to skip only when **both**
  `spot_pos[idx][2] == 0 and spot_pos[idx][3] == 0`, and guard the
  `calculate_*_coverage` calls on the spot radius so a zero spot radius gives
  `asp = 0`. Facular coverage formula stays `max(0, cov(rad_fac) - cov(rad_spot))`
  but use **this spot's** spot coverage, not the running `asp` total (fix the
  pre-existing undercount for overlapping spots).

### B4. Results
Add `self.results['ff_fc']` (already there) and store per-star mean facular
and spot filling factors in the config dict written by `create_data.simulate_one`
(`activity_params`: `facula_q`, `facula_contrast_scale`, `facula_lifetime_ratio`,
`mean_ff_sp`, `mean_ff_fc`).

### Tests for Part B (`tests/test_faculae.py`)
Use `test_spectra.conf`-style small config, `use_phoenix_limb_darkening=1`,
solar parameters (5800 K, logg 4.4, feh 0, P = 25 d, R = 1), a **single**
hand-placed spot via `star.spot_map`, `spots_evo_law='constant'`, 1000 time
points over one rotation, no noise, Kepler filter.
* T-B0 regression: `Q = 0` → light curve identical (`np.allclose(atol=1e-12)`)
  to the output of the current `master` for the same spot map. Generate the
  reference once on `master` and store it as `tests/data/ref_lc_noQ.npy`.
* T-B1 limb brightening: spot radius 0.01 deg (negligible), `rad_fac` set
  directly to 10 deg (facula-only), latitude 0, `i = 90`. Expect: flux
  above the immaculate level whenever the feature is visible, with **two
  maxima** per rotation (near each limb crossing) and a local minimum at
  central-meridian crossing. Check: `flux[phase≈0] < max(flux)` and the
  maximum occurs at `|phase| > 0.15` rotations from central crossing.
* T-B2 spot + facula, `Q = 3`, spot 5 deg, latitude 0, `i = 90`: the light
  curve shows a dip at centre crossing flanked by brightening shoulders;
  `min(flux) < 1 < max(flux)`.
* T-B3 inclination response: same feature at latitude 0, compare `i = 90`
  and `i = 30`. Define spot-only curve (run with `Q=0`) and facula-only
  curve (B1 setup). Expect the ratio
  `ptp(facula_only)/ptp(spot_only)` to be **larger** at `i = 30` than at
  `i = 90` (faculae are limb-weighted; a low-latitude feature seen at low
  inclination sits nearer the limb). Report both numbers in the test log.
* T-B4 lifetime: gaussian law, `facula_lifetime_ratio = 3`, one spot emerging
  at t = 0 with `tau_decay = 20` d: at t = 60 d `ff_sp < 0.01*ff_sp(peak)`
  while `ff_fc > 0.1*ff_fc(peak)`.
* T-B5 energy sanity: with `Q = 3` the brightening never exceeds
  `afc_max * (brigh_hot - brigh_ph)` summed over the disk (no negative
  photosphere fraction; `aph >= 0` for all pixels — assert inside the loop
  under a debug flag or check `ff_ph + ff_sp + ff_fc + ff_pl == 100 ± 1e-6`).

---------------------------------------------------------------------------

## Part C — population priors for faculae (`create_data.py`)

New columns in `generate_simdata` (all per star, exposed in
`simulation_properties.csv` so a hierarchical model can later re-weight them):
* `Facula Q`: `10**truncnorm(mean=log10(3), sd=0.4, lo=log10(0.3), hi=log10(20))`.
  Solar area ratio ≈ 3 (Chapman et al. 1997); Shapiro et al. 2014 find Q
  decreasing with activity, so **optionally** (flag `--q_activity_dep`)
  multiply by `(Activity Rate)**(-0.3)` clipped to [0.3, 20].
* `Facula Contrast Scale`: `truncnorm(1.0, 0.3, 0.3, 2.0)`.
* `Facula Lifetime Ratio`: `10**uniform(log10(1.5), log10(5))`.
Pass them through `simulate_one` → `set_stellar_parameters` → config dict.
Add them to `plot_distributions` automatically (it iterates columns).

Test T-C1: `generate_simdata(N=1000)` has the three columns within bounds and
`dataset_distributions.png` renders.

---------------------------------------------------------------------------

## Part D — faculae in the spectra (consistency, lower priority)

`spectra.create_observed_spectra_kernel`, lines ~486-520, lumps faculae into
the spot kernel (`visible_spot = (asp + afc) * ...`). Change to three kernels:
* `visible_fac = afc * (1 - apl) * area_pixels` → `kernel_fc`.
* Facular intensity at PHOENIX angle `mu_idx`:
  `I_fc = photo_flux[mu_idx] + w(mu_sini[mu_idx]) * (hot_flux[mu_idx] - photo_flux[mu_idx])`
  with `w = facula_dT_mu(mu, scale)/DT_REF`; pass `hot_flux` (= `star._hot_flux`)
  as a new keyword argument (default `None` → old behaviour).
* Convolve and add like the spot term.
Also update `create_observed_spectra` (fallback path) to accept `ff_fc` and
blend with the disk-averaged facular spectrum, or document that the fallback
ignores faculae. `check_flux_consistency.py` must still give ratio 1.000.

Test T-D1: coverage array with `afc = 1` everywhere → spectrum lies between
the photosphere and the `+DT_REF` spectrum at every wavelength and is closer
to the photosphere (mean weight ≈ 0.3).

---------------------------------------------------------------------------

## Part E — documentation and a smoke dataset
* README: replace "faculae currently not supported" with a short description
  of Q, contrast scale, lifetime ratio, and the Meunier CLV law; mention the
  per-star seed.
* JOSS `paper.md`: one paragraph on faculae.
* Run `python -m Dynamo.create_data --num_simulations 200 --ndays 1000 --n_cpu 4 --dataset_dir dataset_faculae_smoke`
  and report: amplitude percentiles (see A5 target), fraction of stars where
  the facular variance exceeds the spot variance (compute from `ff_sp`,
  `ff_fc` time series or by re-running with `Q=0`), split by `Period < 15 d`
  vs `> 25 d`. Expected qualitatively: faculae-dominated fraction higher for
  slow rotators (Montet et al. 2017). Put the numbers in `docs/faculae_smoke.md`.

---------------------------------------------------------------------------

## Acceptance checklist
- [ ] T-A1..A4, T-B0..B5, T-C1, T-D1 pass; existing `tests/test_period_recovery_suite.py`
      and `tests/test_multi_planet.py` still pass.
- [ ] `Q = 0` reproduces master light curves exactly (T-B0).
- [ ] Runtime per star (1000 d, 48/day, Kepler only) increases by less than 30%
      with faculae on (one extra PHOENIX interpolation + larger visible radii).
- [ ] `simulation_properties.csv` carries: Seed, Has Dynamo, Spot Contrast Scale,
      Spot Max Area, Facula Q, Facula Contrast Scale, Facula Lifetime Ratio.
- [ ] Smoke dataset numbers reported in `docs/faculae_smoke.md`.

## Known limitations to state in the docs (do not try to fix now)
* Facular CLV law is solar; Teff dependence enters only via `Facula Contrast Scale`.
* Faculae are annuli around spots; no independent network component.
* Linear-in-T blending between `T_ph` and `T_ph + 300 K` PHOENIX models.
* Spot lifetime is independent of spot size.
