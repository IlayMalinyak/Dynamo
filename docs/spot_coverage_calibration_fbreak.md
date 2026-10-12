# Spot-coverage calibration readout

R_per of clean 1000-d light curves (faculae on, no noise). Main sequence, Teff < 6500 K.

|  | arm | N (MS, <6500 K) | R_per 5% [ppt] | 50% | 95% | bimodality (log R_per) | spots median | spots max | boosted | rho(R_per, Ro) | rho(R_per, area) | plateau 0.2-0.5 [ppt] | slope 0.5-0.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dataset_cov_legacy | 133 | 1.02 | 5.12 | 21.2 | 0.43 | 272 | 500 | 0% | -0.30 | +0.84 | 7.8 | +0.2 |
| 1 | dataset_cov_act_fb0300 | 133 | 0.24 | 3.84 | 19.9 | 0.49 | 1304 | 13041 | 3% | -0.89 | +0.83 | 7.1 | -3.1 |
| 2 | dataset_cov_act_fb0600 | 133 | 0.24 | 6.44 | 27.5 | 0.55 | 2420 | 17587 | 9% | -0.89 | +0.82 | 10.7 | -4.2 |
| 3 | dataset_cov_act_fb1200 | 133 | 0.24 | 9.18 | 39.6 | 0.58 | 4267 | 27065 | 17% | -0.90 | +0.79 | 15.1 | -5.0 |
| 4 | McQuillan 2014 / Masuda 2022 |  | 0.95 | 5.6 | 22.7 |  |  |  |  |  |  | 10.8 | -4.6 |

R_per median vs Ro/Ro_sun, 0.9-1.1 M_sun (Masuda 2022: ~10.8 ppt plateau at 0.2-0.5, slope -4.6 above 0.5). Plateau and slope columns above: 4000-6250 K dwarfs.

|  | Ro/Ro_sun | dataset_cov_act_fb0300 | dataset_cov_act_fb0600 | dataset_cov_act_fb1200 | dataset_cov_legacy |
|---|---|---|---|---|---|
| 0 | 0.05-0.1 | 11.57 | 13.19 | 19.07 | 2.68 |
| 1 | 0.1-0.2 | - | - | - | - |
| 2 | 0.2-0.4 | 7.28 | 11.62 | 18.38 | 4.36 |
| 3 | 0.4-0.6 | 6.19 | 8.89 | 13.13 | 6.63 |
| 4 | 0.6-0.8 | 2.02 | 1.87 | 2.29 | 7.31 |
| 5 | 0.8-1.0 | 0.57 | 0.73 | 0.63 | 2.85 |
| 6 | 1.0-1.5 | 0.43 | 0.43 | 0.43 | 2.54 |
| 7 | 1.5-3.0 | 0.91 | 0.91 | 0.91 | 25.50 |
