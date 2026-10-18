# Phase 2: Model Comparison & Validation Report

**Dataset:** `data/processed/clean.csv` (35,058 verified listings across 81 cities)  
**Cross-Validation:** 5-Fold Cross Validation with Leak-Free Featurization Pipelines  
**Evaluation Target:** $\log(\text{price})$ (evaluated on both log scale and reconstructed Rupee scale)  

---

## 1. 5-Fold Cross-Validation Performance Comparison

| Model | $R^2$ (Log Scale) | MAE (Log) | RMSE (Log) | $R^2$ (Rupee Scale) | MAE (Rupees) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge (Baseline)** | 0.7923 +/- 0.0066 | 0.2766 | 0.3628 | -3.4911 | 39.60 Lac | **29.57%** |
| **Random Forest** | 0.8272 +/- 0.0045 | 0.2532 | 0.3309 | 0.7697 | 28.57 Lac | **26.16%** |
| **XGBoost** | 0.8463 +/- 0.0048 | 0.2411 | 0.3120 | 0.7697 | 27.88 Lac | **24.86%** |
| **LightGBM** | 0.8525 +/- 0.0043 | 0.2353 | 0.3057 | 0.7898 | 27.20 Lac | **24.29%** |

---

## 2. Key Findings & Champion Selection

- **Champion Model:** **LightGBM**
- **Margin Over Ridge Baseline:**
  - Ridge baseline achieved $R^2 = 0.7923$ with a MAPE of 29.57%.
  - **LightGBM achieved $R^2 = 0.8525$ with a MAPE of 24.29%**, cutting the average prediction error significantly.
- **Tree Ensembles vs. Linear Models:**
  - Non-linear interactions between `area_sqft`, `location`, `bhk`, and `floor_no` are captured naturally by tree ensembles without manual interaction term engineering.
  - LightGBM and XGBoost offer ultra-fast inference suitable for the sub-50ms Flask web API response SLA.

---

## 3. Uncertainty & Price Range Estimation (Quantile Regression)

- Trained LightGBM Quantile models at **$\alpha = 0.10$** (10th percentile / lower bound) and **$\alpha = 0.90$** (90th percentile / upper bound).
- **Empirical Coverage on Unseen Test Split:** **77.58%** (Target: 80%).
- **UI Impact:** The web frontend can now show a realistic confidence interval (e.g., ₹62 Lac – ₹74 Lac, median ₹68 Lac) instead of a brittle single-point prediction.

---

## 4. Slice-Based Error Analysis

### Performance by Price Band
| Price Band | Count | Median MAPE (%) | Mean MAPE (%) | MAE (₹ Lakhs) |
| :--- | :---: | :---: | :---: | :---: |
| < 50 Lac | 11,520 | 19.55% | 27.17% | 8.97 Lac |
| 50 Lac - 1.5 Cr | 17,218 | 17.43% | 22.08% | 19.17 Lac |
| 1.5 Cr - 3 Cr | 4,324 | 20.37% | 24.28% | 50.79 Lac |
| > 3 Cr | 1,996 | 24.63% | 26.73% | 150.59 Lac |

### Performance across Top Metros
| City | Listings | Mean MAPE (%) | Median MAPE (%) | MAE (₹ Lakhs) |
| :--- | :---: | :---: | :---: | :---: |
| Faridabad | 2,074 | 18.62% | 15.06% | 16.17 Lac |
| Gurgaon | 1,990 | 27.19% | 21.58% | 67.24 Lac |
| New-Delhi | 1,974 | 30.25% | 21.58% | 64.19 Lac |
| Greater-Noida | 1,588 | 23.22% | 18.59% | 19.56 Lac |
| Ahmedabad | 1,529 | 23.51% | 17.82% | 29.21 Lac |
| Vadodara | 1,347 | 22.28% | 17.65% | 11.69 Lac |
| Pune | 1,341 | 23.43% | 17.85% | 24.70 Lac |
| Chennai | 1,319 | 30.49% | 24.91% | 32.71 Lac |
| Kolkata | 1,267 | 30.28% | 23.70% | 28.46 Lac |
| Hyderabad | 1,252 | 20.55% | 16.51% | 27.22 Lac |
| Bangalore | 1,219 | 28.18% | 21.83% | 40.01 Lac |
| Jaipur | 1,138 | 22.80% | 18.61% | 13.72 Lac |

---

## 5. Residual Diagnostics
Residual distribution, actual-vs-predicted scatter, and error distributions are visualized in `reports/residual_analysis.png`.
