# Model inputs: pandas reference table vs Spark table

Rows compared (eval-eligible train/val/test rows used by the models): **160,580**; input columns: **66** (Model B; Model A drops the rain columns).

## Differing input columns

| column | rows that differ | share of rows |
|---|---:|---:|
| `gw_max_to_date_m` | 20,849 | 12.98% |
| `gw_max_to_date_m_rz` | 20,849 | 12.98% |
| `gw_min_to_date_m` | 20,849 | 12.98% |
| `gw_min_to_date_m_rz` | 20,849 | 12.98% |

- Other input columns that differ: **0**
- Sequence-tensor cells that differ: **0**
- Target (`anomaly_z_capped`) values that differ: **0**
- Stress labels that differ: **0**; split assignments that differ: **0**

## First 80 differing rows of `gw_max_to_date_m` (of 20,849)

| well_id | period | pandas | Spark |
|---|---|---:|---:|
| W0009aafbe6 | Jan-07 | nan | 10.3000 |
| W0009aafbe6 | Nov-12 | nan | 11.1500 |
| W0009aafbe6 | Aug-16 | nan | 12.3800 |
| W0009aafbe6 | Aug-18 | nan | 12.3800 |
| W0009aafbe6 | Jan-19 | nan | 12.3800 |
| W0009aafbe6 | Aug-21 | nan | 12.3800 |
| W002dcc6c18 | Nov-12 | nan | 9.8800 |
| W002dcc6c18 | Jan-19 | nan | 9.8800 |
| W002dcc6c18 | Nov-20 | nan | 9.8800 |
| W002dcc6c18 | Aug-21 | nan | 9.8800 |
| W002dcc6c18 | Nov-22 | nan | 9.8800 |
| W00405c4c51 | Nov-12 | nan | 7.1200 |
| W00405c4c51 | Aug-13 | nan | 7.1200 |
| W00405c4c51 | Aug-14 | nan | 7.1200 |
| W00405c4c51 | Aug-17 | nan | 7.1200 |
| W00405c4c51 | Aug-21 | nan | 7.2700 |
| W00405c4c51 | Aug-22 | nan | 7.2700 |
| W004fd349e7 | Nov-12 | nan | 14.6500 |
| W004fd349e7 | Aug-16 | nan | 14.6500 |
| W004fd349e7 | Aug-21 | nan | 14.6500 |
| W004fd349e7 | Aug-22 | nan | 14.6500 |
| W0050c3dc79 | Jan-10 | nan | 8.3500 |
| W0050c3dc79 | Nov-12 | nan | 8.3500 |
| W0050c3dc79 | Nov-20 | nan | 8.3500 |
| W0050c3dc79 | Aug-21 | nan | 8.3500 |
| W00607b62a5 | Aug-05 | nan | 8.3100 |
| W00607b62a5 | Aug-06 | nan | 8.3100 |
| W00607b62a5 | Aug-07 | nan | 8.3100 |
| W00607b62a5 | Aug-08 | nan | 8.3100 |
| W00607b62a5 | Aug-09 | nan | 8.3100 |
| W00607b62a5 | Aug-10 | nan | 8.3100 |
| W00607b62a5 | Aug-11 | nan | 8.3100 |
| W00607b62a5 | Nov-12 | nan | 8.3100 |
| W00607b62a5 | Aug-13 | nan | 8.3100 |
| W00607b62a5 | Aug-14 | nan | 8.3100 |
| W00607b62a5 | Aug-15 | nan | 8.3100 |
| W00607b62a5 | Aug-16 | nan | 8.3100 |
| W00607b62a5 | Aug-17 | nan | 9.6000 |
| W00607b62a5 | Aug-18 | nan | 9.6000 |
| W00607b62a5 | Aug-19 | nan | 9.6000 |
| W00607b62a5 | Nov-20 | nan | 9.6000 |
| W00607b62a5 | Aug-21 | nan | 9.6000 |
| W00607b62a5 | Aug-22 | nan | 9.6000 |
| W00699d8009 | Nov-11 | nan | 12.6000 |
| W00699d8009 | Nov-12 | nan | 12.6000 |
| W00699d8009 | Aug-18 | nan | 12.6000 |
| W00699d8009 | Nov-21 | nan | 14.0000 |
| W00699d8009 | May-22 | nan | 14.0000 |
| W006cbe0f0a | Aug-05 | nan | 6.2900 |
| W006cbe0f0a | Aug-06 | nan | 6.2900 |
| W006cbe0f0a | Aug-07 | nan | 6.7700 |
| W006cbe0f0a | Aug-08 | nan | 6.7700 |
| W006cbe0f0a | Nov-09 | nan | 6.7700 |
| W006cbe0f0a | Aug-10 | nan | 6.7700 |
| W006cbe0f0a | Aug-11 | nan | 6.7700 |
| W006cbe0f0a | Nov-12 | nan | 6.7700 |
| W006cbe0f0a | Aug-13 | nan | 6.7700 |
| W006cbe0f0a | Aug-14 | nan | 6.7700 |
| W006cbe0f0a | Aug-15 | nan | 6.7700 |
| W006cbe0f0a | Aug-16 | nan | 6.7700 |
| W006cbe0f0a | Aug-17 | nan | 6.7700 |
| W006cbe0f0a | Aug-18 | nan | 6.7700 |
| W006cbe0f0a | Jan-19 | nan | 6.7700 |
| W006cbe0f0a | Aug-19 | nan | 6.7700 |
| W006cbe0f0a | Nov-20 | nan | 6.7700 |
| W006cbe0f0a | Aug-21 | nan | 6.7700 |
| W006cbe0f0a | Aug-22 | nan | 6.7700 |
| W007469ed87 | Aug-05 | nan | 4.8000 |
| W007469ed87 | Nov-12 | nan | 5.2400 |
| W007469ed87 | Aug-13 | nan | 5.2400 |
| W007469ed87 | Nov-20 | nan | 5.6200 |
| W007469ed87 | Aug-21 | nan | 5.6200 |
| W007469ed87 | May-22 | nan | 5.6200 |
| W007a25219d | Nov-12 | nan | 3.3500 |
| W007a25219d | Aug-15 | nan | 3.3500 |
| W007a25219d | Jan-19 | nan | 3.3500 |
| W007a25219d | Aug-21 | nan | 4.1900 |
| W0080669cf4 | Nov-12 | nan | 8.7400 |
| W0080669cf4 | Nov-13 | nan | 8.7400 |
| W0080669cf4 | Nov-20 | nan | 10.0800 |

## First 80 differing rows of `gw_min_to_date_m` (of 20,849)

| well_id | period | pandas | Spark |
|---|---|---:|---:|
| W0009aafbe6 | Jan-07 | nan | 1.1900 |
| W0009aafbe6 | Nov-12 | nan | 0.8200 |
| W0009aafbe6 | Aug-16 | nan | 0.8200 |
| W0009aafbe6 | Aug-18 | nan | 0.8200 |
| W0009aafbe6 | Jan-19 | nan | 0.8200 |
| W0009aafbe6 | Aug-21 | nan | 0.8200 |
| W002dcc6c18 | Nov-12 | nan | 0.6400 |
| W002dcc6c18 | Jan-19 | nan | 0.6400 |
| W002dcc6c18 | Nov-20 | nan | 0.6400 |
| W002dcc6c18 | Aug-21 | nan | 0.6400 |
| W002dcc6c18 | Nov-22 | nan | 0.1500 |
| W00405c4c51 | Nov-12 | nan | 1.3600 |
| W00405c4c51 | Aug-13 | nan | 1.3600 |
| W00405c4c51 | Aug-14 | nan | 1.3600 |
| W00405c4c51 | Aug-17 | nan | 1.3600 |
| W00405c4c51 | Aug-21 | nan | 1.3600 |
| W00405c4c51 | Aug-22 | nan | 1.3600 |
| W004fd349e7 | Nov-12 | nan | 0.2500 |
| W004fd349e7 | Aug-16 | nan | 0.2500 |
| W004fd349e7 | Aug-21 | nan | 0.2500 |
| W004fd349e7 | Aug-22 | nan | 0.2500 |
| W0050c3dc79 | Jan-10 | nan | 2.2500 |
| W0050c3dc79 | Nov-12 | nan | 2.2500 |
| W0050c3dc79 | Nov-20 | nan | 2.0700 |
| W0050c3dc79 | Aug-21 | nan | 1.5700 |
| W00607b62a5 | Aug-05 | nan | 3.0300 |
| W00607b62a5 | Aug-06 | nan | 3.0300 |
| W00607b62a5 | Aug-07 | nan | 3.0300 |
| W00607b62a5 | Aug-08 | nan | 3.0300 |
| W00607b62a5 | Aug-09 | nan | 3.0300 |
| W00607b62a5 | Aug-10 | nan | 3.0300 |
| W00607b62a5 | Aug-11 | nan | 3.0300 |
| W00607b62a5 | Nov-12 | nan | 3.0300 |
| W00607b62a5 | Aug-13 | nan | 3.0300 |
| W00607b62a5 | Aug-14 | nan | 3.0300 |
| W00607b62a5 | Aug-15 | nan | 3.0300 |
| W00607b62a5 | Aug-16 | nan | 3.0300 |
| W00607b62a5 | Aug-17 | nan | 3.0300 |
| W00607b62a5 | Aug-18 | nan | 3.0300 |
| W00607b62a5 | Aug-19 | nan | 3.0300 |
| W00607b62a5 | Nov-20 | nan | 3.0300 |
| W00607b62a5 | Aug-21 | nan | 3.0300 |
| W00607b62a5 | Aug-22 | nan | 3.0300 |
| W00699d8009 | Nov-11 | nan | 1.5300 |
| W00699d8009 | Nov-12 | nan | 1.5300 |
| W00699d8009 | Aug-18 | nan | 1.5300 |
| W00699d8009 | Nov-21 | nan | 1.5300 |
| W00699d8009 | May-22 | nan | 1.5300 |
| W006cbe0f0a | Aug-05 | nan | 0.9000 |
| W006cbe0f0a | Aug-06 | nan | 0.9000 |
| W006cbe0f0a | Aug-07 | nan | 0.1000 |
| W006cbe0f0a | Aug-08 | nan | 0.1000 |
| W006cbe0f0a | Nov-09 | nan | 0.1000 |
| W006cbe0f0a | Aug-10 | nan | 0.1000 |
| W006cbe0f0a | Aug-11 | nan | 0.1000 |
| W006cbe0f0a | Nov-12 | nan | 0.1000 |
| W006cbe0f0a | Aug-13 | nan | 0.1000 |
| W006cbe0f0a | Aug-14 | nan | 0.1000 |
| W006cbe0f0a | Aug-15 | nan | 0.1000 |
| W006cbe0f0a | Aug-16 | nan | 0.1000 |
| W006cbe0f0a | Aug-17 | nan | 0.1000 |
| W006cbe0f0a | Aug-18 | nan | 0.1000 |
| W006cbe0f0a | Jan-19 | nan | 0.0800 |
| W006cbe0f0a | Aug-19 | nan | 0.0800 |
| W006cbe0f0a | Nov-20 | nan | 0.0800 |
| W006cbe0f0a | Aug-21 | nan | 0.0800 |
| W006cbe0f0a | Aug-22 | nan | 0.0800 |
| W007469ed87 | Aug-05 | nan | 2.3000 |
| W007469ed87 | Nov-12 | nan | 1.7000 |
| W007469ed87 | Aug-13 | nan | 1.7000 |
| W007469ed87 | Nov-20 | nan | 1.7000 |
| W007469ed87 | Aug-21 | nan | 1.1500 |
| W007469ed87 | May-22 | nan | 1.1500 |
| W007a25219d | Nov-12 | nan | 0.2900 |
| W007a25219d | Aug-15 | nan | 0.2900 |
| W007a25219d | Jan-19 | nan | 0.2900 |
| W007a25219d | Aug-21 | nan | 0.0800 |
| W0080669cf4 | Nov-12 | nan | 1.4800 |
| W0080669cf4 | Nov-13 | nan | 1.4800 |
| W0080669cf4 | Nov-20 | nan | 1.3800 |

