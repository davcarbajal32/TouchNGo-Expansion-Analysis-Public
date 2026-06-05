# Touch N Go Soccer — Expansion Impact Analysis

A senior capstone project (Cal Poly Pomona, Computer Science, Spring 2026) analyzing
customer behavior across Touch N Go Soccer's two Southern California facilities and
forecasting the impact of opening a potential third location in Laguna Niguel.

## Note on data
This repository contains code only. All datasets, the data-collection scraper, and the
interactive dashboards are kept private because they contain real customer records and
confidential business information. As a result, the scripts here are not runnable end to
end — they are published to demonstrate methodology, structure, and modeling approach.

## Project goals
1. Analyze customer behavior, attendance, and geography at the existing locations.
2. Identify the factors that drive retention and membership conversion.
3. Forecast the impact of an expansion, separating genuine new growth from demand
   shifted away from existing facilities.

## Approach
Internal customer data (~132K raw visits, cleaned to ~126K) was collected from a
third-party platform with no accessible API via a custom Selenium scraper, then
engineered into three datasets at the geographic (ZIP), customer, and visit levels,
combined with public U.S. Census demographics and computed drive times.

Three complementary models were used:
- **Huff Model** — a spatial choice model estimating the probability a customer chooses
  a given location based on drive time.
- **Logistic Regression** — interpretable prediction of churn and membership conversion.
- **Random Forest** — captures nonlinear patterns in churn, conversion, and post-visit
  return, and powers a ZIP-level demand forecast weighted by Huff probabilities.

**Leakage discipline:** all behavioral models were restricted to features knowable within
a customer's first 30 days. Outcome-encoding features (total visits, days since last
visit, average visits per month) were deliberately excluded so the models predict future
behavior rather than memorize it.

## Results
- Membership conversion (Random Forest): ROC-AUC 0.909 (0.947 Corona, 0.935 Tustin)
- Post-visit return (Random Forest): ROC-AUC 0.813
- Churn (Random Forest): ROC-AUC 0.764 — consistently outperformed logistic regression
- ZIP-level demand forecast (Random Forest): cross-validated R² 0.770

**Headline finding:** of projected Year-1 demand at the new location, ~71% is genuinely
new to Touch N Go and ~29% is redistributed from existing facilities (concentrated almost
entirely at Tustin). Drive time was the dominant predictor across every model. The
cannibalization estimate was validated two independent ways (Huff baseline and Random
Forest demand model) plus a sensitivity sweep.

## Repository structure
- `data_collection_pipeline/` — dataset cleaning and feature engineering
- `analysis/` — Goal 1 descriptive and geographic analysis
- `model_building/` — Goal 2 and Goal 3 model code
- `charts/` — selected aggregate result visualizations

## Tech
Python, pandas, scikit-learn, Selenium (scraper, kept private), OSRM for routing,
U.S. Census ACS API.
