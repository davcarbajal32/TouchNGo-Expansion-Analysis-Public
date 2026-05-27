# Model A - Goal 2
# Predicts whether a customer has churned (no visit in 90+ days)
# Uses early-behavior features from first 30 days only to avoid leakage
# Builds 3 versions: combined, Corona only, Tustin only

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


# load data
print("=" * 60)
print("MODEL A - Predict churn (no visit in 90+ days)")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
d1 = pd.read_csv(os.path.join(script_dir, "dataset1_final.csv"))
d2 = pd.read_csv(os.path.join(script_dir, "dataset2_final.csv"))
d3 = pd.read_csv(os.path.join(script_dir, "dataset3_final.csv"))
d3['date'] = pd.to_datetime(d3['date'])

print("DS1 rows:", len(d1))
print("DS2 rows:", len(d2))
print("DS3 rows:", len(d3))


# fix coach name typo
d3['coach'] = d3['coach'].replace({'Yannelli Chavez': 'Yanelli Chavez'})


# use only signed-in visits for feature engineering and target
# absences and late cancels are not real attendance
d3_signed = d3[d3['status'] == 'Signed in'].copy()

data_end = d3['date'].max()
cutoff_30d = data_end - pd.Timedelta(days=30)

print("Data end:", data_end.date())
print("Cohort cutoff (first visit must be on or before):", cutoff_30d.date())


# find first visit per (client_id, facility) using composite key
first_visits = (d3_signed.sort_values(['client_id', 'facility', 'date'])
                          .groupby(['client_id', 'facility']).first().reset_index())
first_visits = first_visits[['client_id', 'facility', 'date',
                              'session_category', 'coach', 'month',
                              'day_of_week']].rename(columns={
    'date': 'first_visit_date',
    'session_category': 'first_session_category',
    'coach': 'first_coach',
    'month': 'first_month',
    'day_of_week': 'first_day_of_week'
})

# cohort: first visit on or after 2022-01-01 (post-COVID) and on or before cutoff
cohort = first_visits[(first_visits['first_visit_date'] >= '2022-01-01') &
                      (first_visits['first_visit_date'] <= cutoff_30d)].copy()
print("Cohort size:", len(cohort))


# engineer early-behavior features from first 30 days
d3_with_first = d3_signed.merge(
    cohort[['client_id', 'facility', 'first_visit_date']],
    on=['client_id', 'facility'], how='inner'
)
d3_with_first['days_from_first'] = (d3_with_first['date'] - d3_with_first['first_visit_date']).dt.days

# visits in first 30 days
v30 = (d3_with_first[d3_with_first['days_from_first'] <= 30]
       .groupby(['client_id', 'facility']).size().reset_index(name='visits_first_30d'))
cohort = cohort.merge(v30, on=['client_id', 'facility'], how='left')
cohort['visits_first_30d'] = cohort['visits_first_30d'].fillna(1).astype(int)

# has_second_visit (any signed-in visit beyond the first)
visits_per_key = d3_with_first.groupby(['client_id', 'facility']).size().reset_index(name='total_signedin')
cohort = cohort.merge(visits_per_key, on=['client_id', 'facility'], how='left')
cohort['has_second_visit'] = (cohort['total_signedin'] >= 2).astype(int)
cohort = cohort.drop(columns=['total_signedin'])

# returned within 7 days of first visit
ret7 = (d3_with_first[(d3_with_first['days_from_first'] >= 1) &
                       (d3_with_first['days_from_first'] <= 7)]
        [['client_id', 'facility']].drop_duplicates())
ret7['returned_within_7_of_first'] = 1
cohort = cohort.merge(ret7, on=['client_id', 'facility'], how='left')
cohort['returned_within_7_of_first'] = cohort['returned_within_7_of_first'].fillna(0).astype(int)

# returned within 30 days of first visit
ret30 = (d3_with_first[(d3_with_first['days_from_first'] >= 1) &
                        (d3_with_first['days_from_first'] <= 30)]
         [['client_id', 'facility']].drop_duplicates())
ret30['returned_within_30_of_first'] = 1
cohort = cohort.merge(ret30, on=['client_id', 'facility'], how='left')
cohort['returned_within_30_of_first'] = cohort['returned_within_30_of_first'].fillna(0).astype(int)

# first visit on weekend
cohort['first_weekend'] = cohort['first_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)


# churn target: last signed-in visit > 90 days before data end
last_visits = (d3_signed.groupby(['client_id', 'facility'])['date'].max().reset_index()
                         .rename(columns={'date': 'last_visit_date'}))
cohort = cohort.merge(last_visits, on=['client_id', 'facility'], how='left')
cohort['days_since_last_visit'] = (data_end - cohort['last_visit_date']).dt.days
cohort['is_churned'] = (cohort['days_since_last_visit'] > 90).astype(int)


# join drive time and zip from DS2 using composite key
d2_join = d2[['client_id', 'facility', 'zip', 'drive_time_to_attended_facility']].drop_duplicates(
    subset=['client_id', 'facility']
)
cohort = cohort.merge(d2_join, on=['client_id', 'facility'], how='left')


# join census demographics from DS1 via ZIP
cohort['zip_int'] = pd.to_numeric(cohort['zip'], errors='coerce')
d1_features = d1[['zip', 'median_household_income', 'youth_pop_5_17',
                   'pct_households_with_children']].rename(columns={'zip': 'zip_int'})
cohort = cohort.merge(d1_features, on='zip_int', how='left')

# impute missing demographic features with facility median
for c in ['median_household_income', 'youth_pop_5_17', 'pct_households_with_children']:
    for fac in cohort['facility'].unique():
        med = cohort.loc[cohort['facility'] == fac, c].median()
        mask = (cohort['facility'] == fac) & cohort[c].isna()
        cohort.loc[mask, c] = med

print("Churn rate:", round(cohort['is_churned'].mean() * 100, 1), "%")


# replace less common coaches with 'Other'
def encode_coach_top_n(df_in, top_n=10):
    top_coaches = df_in['first_coach'].value_counts().head(top_n).index.tolist()
    df_out = df_in.copy()
    df_out['first_coach'] = df_out['first_coach'].where(df_out['first_coach'].isin(top_coaches), 'Other')
    return df_out, top_coaches


# main model builder
def build_models(df_subset, version_name, include_facility=True):
    print("=" * 60)
    print("VERSION:", version_name)
    print("=" * 60)
    print("Rows:", len(df_subset))
    print("Churn rate:", round(df_subset['is_churned'].mean() * 100, 1), "%")

    df_subset, top_coaches = encode_coach_top_n(df_subset, top_n=10)
    print("Top first-visit coaches:", top_coaches[:5], "... plus Other")

    feature_cols = [
        'visits_first_30d', 'has_second_visit',
        'returned_within_7_of_first', 'returned_within_30_of_first',
        'first_session_category', 'first_coach', 'first_month', 'first_weekend',
        'drive_time_to_attended_facility',
        'median_household_income', 'youth_pop_5_17', 'pct_households_with_children'
    ]
    if include_facility:
        feature_cols.insert(0, 'facility')

    # apply log transform to skewed numeric features
    df_subset = df_subset.copy()
    for c in ['visits_first_30d', 'drive_time_to_attended_facility',
              'median_household_income', 'youth_pop_5_17']:
        df_subset[c] = np.log1p(df_subset[c])

    X = df_subset[feature_cols].copy()
    y = df_subset['is_churned'].copy()

    categorical = [c for c in feature_cols if pd.api.types.is_string_dtype(X[c])]
    X = pd.get_dummies(X, columns=categorical, drop_first=False)
    X = X.astype({c: 'float' for c in X.columns if X[c].dtype == 'bool'})
    print("Features after encoding:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("Train:", len(X_train), "Test:", len(X_test))

    metrics = {'version': version_name, 'n_rows': len(df_subset), 'n_features': X.shape[1]}

    # logistic regression
    print("--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_pr_auc = average_precision_score(y_test, lr_proba)
    print("ROC-AUC:", round(lr_auc, 4))
    print("PR-AUC:", round(lr_pr_auc, 4))
    print(classification_report(y_test, lr_pred, digits=3))

    coef_df = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_[0]})
    coef_df = coef_df.sort_values('coef', key=abs, ascending=False)
    print("Top 15 LR coefficients:")
    print(coef_df.head(15).to_string(index=False))

    metrics['lr_roc_auc'] = lr_auc
    metrics['lr_pr_auc'] = lr_pr_auc

    # random forest
    print("--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_pr_auc = average_precision_score(y_test, rf_proba)
    print("ROC-AUC:", round(rf_auc, 4))
    print("PR-AUC:", round(rf_pr_auc, 4))
    print(classification_report(y_test, rf_pred, digits=3))

    fi_df = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print("Top 15 RF feature importances:")
    print(fi_df.head(15).to_string(index=False))

    metrics['rf_roc_auc'] = rf_auc
    metrics['rf_pr_auc'] = rf_pr_auc

    return metrics, lr, rf, coef_df, fi_df


# run all three versions
all_results = {}

all_results['combined'] = build_models(cohort.copy(), "COMBINED", include_facility=True)
all_results['corona'] = build_models(cohort[cohort['facility'] == 'Corona'].copy(), "CORONA ONLY", include_facility=False)
all_results['tustin'] = build_models(cohort[cohort['facility'] == 'Tustin'].copy(), "TUSTIN ONLY", include_facility=False)


# final summary
print("=" * 60)
print("MODEL A - CHURN PREDICTION COMPARISON")
print("=" * 60)

summary = pd.DataFrame([
    {
        'Version': r[0]['version'],
        'Rows': r[0]['n_rows'],
        'Features': r[0]['n_features'],
        'LR ROC-AUC': round(r[0]['lr_roc_auc'], 4),
        'LR PR-AUC': round(r[0]['lr_pr_auc'], 4),
        'RF ROC-AUC': round(r[0]['rf_roc_auc'], 4),
        'RF PR-AUC': round(r[0]['rf_pr_auc'], 4),
    }
    for r in all_results.values()
])
print(summary.to_string(index=False))

print("=" * 60)
print("DONE")
print("=" * 60)
