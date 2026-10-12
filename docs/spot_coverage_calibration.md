# Spot-coverage calibration readout

R_per of clean 1000-d light curves (faculae on, no noise). Main sequence, Teff < 6500 K.

|  | arm | N (MS, <6500 K) | R_per 5% [ppt] | 50% | 95% | bimodality (log R_per) | spots median | spots max | boosted | rho(R_per, Ro) | rho(R_per, area) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dataset_cov_legacy | 150 | 1.03 | 6.59 | 23.8 | 0.42 | 270 | 500 | 0% | -0.25 | +0.75 |
| 1 | dataset_cov_f0030 | 150 | 1.04 | 4.66 | 16.3 | 0.30 | 210 | 5591 | 0% | -0.59 | +0.40 |
| 2 | dataset_cov_f0100 | 150 | 2.16 | 9.94 | 28.9 | 0.37 | 674 | 11600 | 1% | -0.56 | +0.49 |
| 3 | dataset_cov_f0290 | 150 | 4.38 | 17.80 | 48.5 | 0.39 | 1946 | 14860 | 6% | -0.53 | +0.50 |
| 4 | McQuillan 2014 |  | 0.95 | 5.6 | 22.7 |  |  |  |  |  |  |

R_per median vs Ro/Ro_sun, 0.9-1.1 M_sun (Masuda 2022: ~10 ppt plateau at 0.2-0.4, steep decline above ~0.4)

|  | Ro/Ro_sun | dataset_cov_f0030 | dataset_cov_f0100 | dataset_cov_f0290 | dataset_cov_legacy |
|---|---|---|---|---|---|
| 0 | 0.05-0.1 | 6.24 | 9.14 | 19.04 | 2.68 |
| 1 | 0.1-0.2 | - | - | - | - |
| 2 | 0.2-0.4 | 4.99 | 10.45 | 18.17 | 4.36 |
| 3 | 0.4-0.6 | 4.57 | 11.03 | 19.79 | 6.63 |
| 4 | 0.6-0.8 | 4.68 | 9.49 | 20.83 | 8.10 |
| 5 | 0.8-1.0 | 2.87 | 6.76 | 11.40 | 2.85 |
| 6 | 1.0-1.5 | 1.29 | 3.81 | 6.23 | 2.54 |
| 7 | 1.5-3.0 | 5.15 | 10.53 | 19.75 | 25.50 |
