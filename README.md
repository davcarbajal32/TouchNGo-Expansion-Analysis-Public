# Touch N Go Soccer — Expansion Impact Analysis

A senior capstone project (Cal Poly Pomona, Computer Science, Spring 2026) analyzing customer behavior across Touch N Go Soccer's two Southern California facilities and forecasting the impact of a potential third location.

## Note on data

This repository contains code only. All datasets, the data-collection scraper, and the interactive dashboards are kept private because they contain real customer records and confidential business information. As a result, the scripts here are not runnable end to end — they are published to demonstrate methodology, structure, and modeling approach.

## Project goals

1. Analyze customer behavior, attendance, and geography at the existing locations.
2. Identify the factors that drive retention and membership conversion.
3. Forecast the impact of an expansion, separating genuine new growth from demand shifted away from existing facilities.

## Approach

Internal customer data was collected from a third-party platform and engineered into three datasets at the geographic, customer, and visit levels, combined with public U.S. Census demographics and computed drive times.

Three complementary models were used:

- Huff Model — a spatial choice model estimating the probability a customer chooses a given location based on travel distance.
- Logistic Regression — interpretable prediction of churn and membership conversion.
- Random Forest — captures nonlinear patterns in churn, conversion, and post-visit return, and powers a ZIP-level demand forecast weighted by Huff probabilities.

## Results

- Churn model: cross-validated ROC-AUC 0.764
- Post-visit return model: ROC-AUC 0.813
- Membership conversion model: ROC-AUC 0.909 (above 0.94 at the individual location level)
- Demand model: explained roughly 77% of the variation in customer demand across areas

## Repository structure

- data_collection_pipeline/ — dataset cleaning and feature engineering
- analysis/ — Goal 1 descriptive and geographic analysis
- model_building/ — Goal 2 and Goal 3 model code
- charts/ — selected aggregate result visualizations

## Tech

Python, pandas, scikit-learn, Selenium (scraper, kept private), OSRM for routing, U.S. Census ACS API.
