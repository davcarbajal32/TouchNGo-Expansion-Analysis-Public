# Model E - Goal 3
# Predicts unique customers per ZIP using demographics and drive time
# Then applies Huff probabilities to forecast Laguna customer counts
# Final step converts customer counts to visit counts using historical rates
#
# Two-step approach:
#   Step 1: Train regression on ZIPs to predict total unique customers
#   Step 2: For each ZIP apply Huff probabilities to allocate predicted
#           customers across Corona/Tustin/Laguna
#   Step 3: Multiply by historical visits-per-customer to get visit counts

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


print("=" * 60)
print("MODEL E - ZIP-Level Customer Demand Forecast")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
d1 = pd.read_csv(os.path.join(script_dir, "dataset1_final.csv"))
d2 = pd.read_csv(os.path.join(script_dir, "dataset2_final.csv"))
d3 = pd.read_csv(os.path.join(script_dir, "dataset3_final.csv"))
d3['date'] = pd.to_datetime(d3['date'])


# build ZIP panel
zip_lookup = d2[['client_id', 'facility', 'zip']].drop_duplicates(subset=['client_id', 'facility'])
d3_z = d3.merge(zip_lookup, on=['client_id', 'facility'], how='left')
d3_z['zip_int'] = pd.to_numeric(d3_z['zip'], errors='coerce')
d3_filt = d3_z[(d3_z['status'] == 'Signed in') &
               (d3_z['date'] >= '2022-01-01')].dropna(subset=['zip_int'])
d3_filt['zip_int'] = d3_filt['zip_int'].astype(int)


# total unique customers per ZIP using composite key
unique_cust = (d3_filt.drop_duplicates(['zip_int', 'client_id', 'facility'])
                       .groupby('zip_int').size().reset_index(name='unique_customers_2022'))

# total visits per ZIP
total_visits = d3_filt.groupby('zip_int').size().reset_index(name='total_visits_2022')


# build the full ZIP feature panel
d1_features = d1[['zip', 'youth_pop_5_17', 'median_household_income', 'total_households',
                  'pct_households_with_children', 'drive_time_to_corona_min',
                  'drive_time_to_tustin_min', 'drive_time_to_laguna_min',
                  'elementary_schools', 'middle_schools', 'high_schools',
                  'total_schools']].rename(columns={'zip': 'zip_int'})

zp = d1_features.merge(unique_cust, on='zip_int', how='left').fillna({'unique_customers_2022': 0})
zp = zp.merge(total_visits, on='zip_int', how='left').fillna({'total_visits_2022': 0})


# engineered feature: minimum drive time to any existing facility
# this is the actual friction a potential customer faces when choosing
# whether to engage with Touch N Go at all
zp['min_drive_time_existing'] = zp[['drive_time_to_corona_min', 'drive_time_to_tustin_min']].min(axis=1)

print("ZIP panel shape:", zp.shape)
print("ZIPs with any customers:", (zp['unique_customers_2022'] > 0).sum())
print("Total customers across ZIPs:", int(zp['unique_customers_2022'].sum()))


# average visits per customer at each facility (for step 3 conversion)
corona_visits_per_cust = d3_filt[d3_filt['facility'] == 'Corona'].shape[0] / \
    d3_filt[d3_filt['facility'] == 'Corona'].drop_duplicates(['client_id', 'facility']).shape[0]
tustin_visits_per_cust = d3_filt[d3_filt['facility'] == 'Tustin'].shape[0] / \
    d3_filt[d3_filt['facility'] == 'Tustin'].drop_duplicates(['client_id', 'facility']).shape[0]

print("Corona visits per customer:", round(corona_visits_per_cust, 1))
print("Tustin visits per customer:", round(tustin_visits_per_cust, 1))


# features for the model (all external, leak-free)
# drive_time_to_laguna_min is INTENTIONALLY EXCLUDED from training
# it will only be used at prediction time
feature_cols = [
    'youth_pop_5_17', 'median_household_income', 'total_households',
    'pct_households_with_children', 'min_drive_time_existing',
    'drive_time_to_corona_min', 'drive_time_to_tustin_min',
    'elementary_schools', 'middle_schools', 'high_schools', 'total_schools'
]


# apply log transform to skewed features
zp_model = zp.copy()
for c in ['youth_pop_5_17', 'median_household_income', 'total_households',
          'min_drive_time_existing', 'drive_time_to_corona_min',
          'drive_time_to_tustin_min']:
    zp_model[c] = np.log1p(zp_model[c])

# target: log-transform unique_customers (heavy right tail)
y = np.log1p(zp_model['unique_customers_2022'])
X = zp_model[feature_cols].copy()

print("Training features:", X.shape[1])
print("Training rows:", len(X))


# train both models with cross-validation since n=443 is small
print("=" * 60)
print("STEP 1: TRAIN CUSTOMER DEMAND MODEL")
print("=" * 60)

# Linear Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

lr_r2_scores = cross_val_score(lr, X_scaled, y, cv=kf, scoring='r2')
lr_mae_scores = -cross_val_score(lr, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')

print("--- Linear Regression (5-fold CV on log target) ---")
print("R^2 mean:", round(lr_r2_scores.mean(), 4), "std:", round(lr_r2_scores.std(), 4))
print("MAE mean:", round(lr_mae_scores.mean(), 4), "std:", round(lr_mae_scores.std(), 4))

# fit on full data for prediction
lr.fit(X_scaled, y)
print("\nLR coefficients (sorted by |magnitude|):")
coef_df = pd.DataFrame({'feature': feature_cols, 'coef': lr.coef_})
coef_df = coef_df.sort_values('coef', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# Random Forest
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_r2_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
rf_mae_scores = -cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')

print("--- Random Forest (5-fold CV on log target) ---")
print("R^2 mean:", round(rf_r2_scores.mean(), 4), "std:", round(rf_r2_scores.std(), 4))
print("MAE mean:", round(rf_mae_scores.mean(), 4), "std:", round(rf_mae_scores.std(), 4))

# fit on full data for prediction
rf.fit(X, y)
print("\nRF feature importances:")
fi_df = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
fi_df = fi_df.sort_values('importance', ascending=False)
print(fi_df.to_string(index=False))


# STEP 2: predict customers per ZIP using current drive times
# this gives the baseline "customer potential" of each ZIP
print("=" * 60)
print("STEP 2: PREDICT BASELINE CUSTOMER POTENTIAL PER ZIP")
print("=" * 60)

lr_pred_log = lr.predict(X_scaled)
rf_pred_log = rf.predict(X)
zp['predicted_customers_lr'] = np.clip(np.expm1(lr_pred_log), 0, None)
zp['predicted_customers_rf'] = np.clip(np.expm1(rf_pred_log), 0, None)

# compare predicted vs actual on full data (training fit, not CV)
print("Training R^2 (LR):", round(r2_score(zp['unique_customers_2022'], zp['predicted_customers_lr']), 4))
print("Training R^2 (RF):", round(r2_score(zp['unique_customers_2022'], zp['predicted_customers_rf']), 4))
print("Training MAE (LR):", round(mean_absolute_error(zp['unique_customers_2022'], zp['predicted_customers_lr']), 2))
print("Training MAE (RF):", round(mean_absolute_error(zp['unique_customers_2022'], zp['predicted_customers_rf']), 2))


# STEP 3: apply Huff probabilities to allocate customers across facilities
print("=" * 60)
print("STEP 3: APPLY HUFF PROBABILITIES TO FORECAST LAGUNA")
print("=" * 60)

# load huff probabilities from Model D
huff_path = os.path.join(script_dir, 'huff_probabilities.csv')
if not os.path.exists(huff_path):
    huff_path = '/tmp/huff_probabilities.csv'
huff = pd.read_csv(huff_path)
huff_cols = ['zip_int', 'p_corona', 'p_tustin', 'p_laguna']
zp = zp.merge(huff[huff_cols], on='zip_int', how='left')

# IMPORTANT - for forecasting Laguna we need a "near-Laguna" predicted demand
# we re-predict customer counts assuming Laguna existed by replacing
# min_drive_time_existing with the minimum of all 3 drive times including Laguna
#
# we CAP this at the training minimum to prevent LR from extrapolating wildly
# the smallest training drive time was approx 2.3 min, so any Laguna drive
# time below that is clipped to that floor
training_min_drive = zp['min_drive_time_existing'].min()
print("Training min drive time floor:", round(training_min_drive, 2))

laguna_min_drive = zp[['drive_time_to_corona_min', 'drive_time_to_tustin_min',
                       'drive_time_to_laguna_min']].min(axis=1).clip(lower=training_min_drive)

zp_laguna = zp_model.copy()
zp_laguna['min_drive_time_existing'] = np.log1p(laguna_min_drive)
X_laguna = zp_laguna[feature_cols]
X_laguna_scaled = scaler.transform(X_laguna)

zp['predicted_customers_lr_with_laguna'] = np.clip(np.expm1(lr.predict(X_laguna_scaled)), 0, None)
zp['predicted_customers_rf_with_laguna'] = np.clip(np.expm1(rf.predict(X_laguna)), 0, None)

# Laguna customers from each ZIP using Huff
zp['laguna_customers_lr'] = zp['predicted_customers_lr_with_laguna'] * zp['p_laguna']
zp['laguna_customers_rf'] = zp['predicted_customers_rf_with_laguna'] * zp['p_laguna']

# Use Tustin's visits-per-customer rate as the Laguna conversion factor
# Tustin is the bigger reference and more representative of a mature operation
zp['laguna_visits_lr'] = zp['laguna_customers_lr'] * tustin_visits_per_cust
zp['laguna_visits_rf'] = zp['laguna_customers_rf'] * tustin_visits_per_cust


# cannibalization estimate using the SAME rule as Model D
#   1. P(Laguna) must be >= 30%
#   2. Laguna must be at most 10 min farther than the existing facility
# this prevents the Huff model from spuriously cannibalizing customers
# who would have no rational reason to switch

THRESHOLD = 0.30
TOLERANCE = 10

zp['corona_can_switch'] = ((zp['p_laguna'] >= THRESHOLD) &
                           (zp['drive_time_to_laguna_min'] <= zp['drive_time_to_corona_min'] + TOLERANCE))
zp['tustin_can_switch'] = ((zp['p_laguna'] >= THRESHOLD) &
                           (zp['drive_time_to_laguna_min'] <= zp['drive_time_to_tustin_min'] + TOLERANCE))

# pull per-facility customer counts from DS3
fac_cust = (d3_filt.drop_duplicates(['zip_int', 'client_id', 'facility'])
                    .groupby(['zip_int', 'facility']).size().reset_index(name='cust'))
fac_cust_wide = fac_cust.pivot(index='zip_int', columns='facility', values='cust').fillna(0).reset_index()
fac_cust_wide.columns.name = None
fac_cust_wide = fac_cust_wide.rename(columns={'Corona': 'cor_cust', 'Tustin': 'tus_cust'})
zp = zp.merge(fac_cust_wide[['zip_int', 'cor_cust', 'tus_cust']], on='zip_int', how='left').fillna(0)

zp['corona_cannibalized_customers'] = np.where(
    zp['corona_can_switch'], zp['cor_cust'] * zp['p_laguna'], 0
)
zp['tustin_cannibalized_customers'] = np.where(
    zp['tustin_can_switch'], zp['tus_cust'] * zp['p_laguna'], 0
)
zp['total_cannibalized_customers'] = zp['corona_cannibalized_customers'] + zp['tustin_cannibalized_customers']
zp['total_cannibalized_visits'] = (
    zp['corona_cannibalized_customers'] * corona_visits_per_cust +
    zp['tustin_cannibalized_customers'] * tustin_visits_per_cust
)


# FINAL FORECAST
# IMPORTANT - LR is unreliable here due to log-space extrapolation
# The log target amplifies LR errors exponentially when predicting outside
# the training distribution. RF handles this gracefully because trees
# cannot extrapolate beyond training values.
# We report both for transparency but trust ONLY the RF predictions.

print("WARNING: LR predictions are unreliable due to extrapolation in log space")
print("The model has never seen drive times this short. Trust RF only.")
print()
print("Total predicted Laguna customers (LR - UNRELIABLE):", round(zp['laguna_customers_lr'].sum(), 0))
print("Total predicted Laguna customers (RF):              ", round(zp['laguna_customers_rf'].sum(), 0))
print("Total predicted Laguna visits (LR - UNRELIABLE):    ", round(zp['laguna_visits_lr'].sum(), 0))
print("Total predicted Laguna visits (RF):                 ", round(zp['laguna_visits_rf'].sum(), 0))
print()
print("Total cannibalized customers:", round(zp['total_cannibalized_customers'].sum(), 0))
print("  From Corona:", round(zp['corona_cannibalized_customers'].sum(), 0))
print("  From Tustin:", round(zp['tustin_cannibalized_customers'].sum(), 0))
print("Total cannibalized visits:   ", round(zp['total_cannibalized_visits'].sum(), 0))
print()
print("NET NEW DEMAND (RF - the trustworthy model):")
print("  Customers:", round(zp['laguna_customers_rf'].sum() - zp['total_cannibalized_customers'].sum(), 0))
print("  Visits:   ", round(zp['laguna_visits_rf'].sum() - zp['total_cannibalized_visits'].sum(), 0))


# top contributing ZIPs
print("=" * 60)
print("TOP 15 ZIPS BY PREDICTED LAGUNA CUSTOMERS (RF model)")
print("=" * 60)
top = zp.sort_values('laguna_customers_rf', ascending=False).head(15)
print(top[['zip_int', 'drive_time_to_laguna_min', 'p_laguna',
           'predicted_customers_rf_with_laguna', 'laguna_customers_rf',
           'laguna_visits_rf', 'total_cannibalized_customers']].round(1).to_string(index=False))


# save forecast for later use
zp.to_csv(os.path.join(script_dir, 'laguna_forecast.csv'), index=False)

print("=" * 60)
print("DONE")
print("=" * 60)