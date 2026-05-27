# Model C - Goal 2
# Predicts whether a customer will return within 30 days after a visit
# Builds 3 versions: combined, Corona only, Tustin only
# Also builds a first-visit-only sub-model in 3 versions

import pandas as pd
import numpy as np
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
print("MODEL C - Predict returned_within_30_days")
print("=" * 60)

d2 = pd.read_csv("dataset2_final.csv")
d3 = pd.read_csv("dataset3_final.csv")
d3['date'] = pd.to_datetime(d3['date'])

print("DS2 rows:", len(d2))
print("DS3 rows:", len(d3))


# fix coach name typo
d3['coach'] = d3['coach'].replace({'Yannelli Chavez': 'Yanelli Chavez'})


# apply filters: 2022 onward, before censoring cutoff, signed in only
data_end = d3['date'].max()
cutoff_30d = data_end - pd.Timedelta(days=30)

print("Data end:", data_end.date())
print("Censoring cutoff:", cutoff_30d.date())

df = d3[
    (d3['date'] >= '2022-01-01') &
    (d3['date'] <= cutoff_30d) &
    (d3['status'] == 'Signed in')
].copy()

print("Visits after filtering:", len(df))


# engineer features using composite key (client_id, facility)
df = df.sort_values(['client_id', 'facility', 'date']).reset_index(drop=True)
df['visit_rank'] = df.groupby(['client_id', 'facility']).cumcount() + 1
df['is_first_visit'] = (df['visit_rank'] == 1).astype(int)

first_visit_dates = df.groupby(['client_id', 'facility'])['date'].min().reset_index()
first_visit_dates = first_visit_dates.rename(columns={'date': 'first_visit_at_facility'})
df = df.merge(first_visit_dates, on=['client_id', 'facility'], how='left')
df['days_since_first_visit'] = (df['date'] - df['first_visit_at_facility']).dt.days


# join drive time from DS2 using composite key
d2_drive = d2[['client_id', 'facility', 'drive_time_to_attended_facility']].drop_duplicates(
    subset=['client_id', 'facility']
)
df = df.merge(d2_drive, on=['client_id', 'facility'], how='left')

# impute missing drive time with facility median
for fac in df['facility'].unique():
    med = df.loc[df['facility'] == fac, 'drive_time_to_attended_facility'].median()
    mask = (df['facility'] == fac) & df['drive_time_to_attended_facility'].isna()
    df.loc[mask, 'drive_time_to_attended_facility'] = med

print("Final dataset:", len(df), "visits,", df.shape[1], "columns")
print("Return rate:", round(df['returned_within_30_days'].mean() * 100, 1), "%")


# replace less common coaches with 'Other'
def encode_coach_top_n(df_in, top_n=10):
    top_coaches = df_in['coach'].value_counts().head(top_n).index.tolist()
    df_out = df_in.copy()
    df_out['coach'] = df_out['coach'].where(df_out['coach'].isin(top_coaches), 'Other')
    return df_out, top_coaches


# main model builder
def build_models(df_subset, version_name, include_facility=True):
    print("=" * 60)
    print("VERSION:", version_name)
    print("=" * 60)
    print("Rows:", len(df_subset))
    print("Return rate:", round(df_subset['returned_within_30_days'].mean() * 100, 1), "%")

    df_subset, top_coaches = encode_coach_top_n(df_subset, top_n=10)
    print("Top coaches:", top_coaches[:5], "... plus Other")

    feature_cols = [
        'session_category', 'coach', 'day_of_week', 'month', 'hour',
        'visit_rank', 'days_since_first_visit', 'is_first_visit',
        'drive_time_to_attended_facility'
    ]
    if include_facility:
        feature_cols.insert(0, 'facility')

    X = df_subset[feature_cols].copy()
    y = df_subset['returned_within_30_days'].copy()

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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_pr_auc = average_precision_score(y_test, lr_proba)

    print("ROC-AUC:", round(lr_auc, 4))
    print("PR-AUC:", round(lr_pr_auc, 4))
    print(classification_report(y_test, lr_pred, digits=3))

    coef_df = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_[0]})
    coef_df = coef_df.sort_values('coef', key=abs, ascending=False)
    print("Top 10 LR coefficients:")
    print(coef_df.head(10).to_string(index=False))

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


# run the general model on all 3 versions
all_results = {}

all_results['combined'] = build_models(df.copy(), "COMBINED", include_facility=True)
all_results['corona'] = build_models(df[df['facility'] == 'Corona'].copy(), "CORONA ONLY", include_facility=False)
all_results['tustin'] = build_models(df[df['facility'] == 'Tustin'].copy(), "TUSTIN ONLY", include_facility=False)


# first-visit-only sub-model
# built because the general model relied too heavily on visit_rank (40% importance)
# this isolates the factors that drive first-impression retention
print("=" * 60)
print("FIRST-VISIT-ONLY SUB-MODEL")
print("=" * 60)


def build_first_visit_models(df_subset, version_name, include_facility=True):
    df_subset = df_subset[df_subset['visit_rank'] == 1].copy()

    print("=" * 60)
    print("VERSION (FIRST VISITS):", version_name)
    print("=" * 60)
    print("Rows:", len(df_subset))
    print("Return rate:", round(df_subset['returned_within_30_days'].mean() * 100, 1), "%")

    df_subset, top_coaches = encode_coach_top_n(df_subset, top_n=10)
    print("Top coaches:", top_coaches[:5], "... plus Other")

    # no tenure features since visit_rank is always 1 for first visits
    feature_cols = [
        'session_category', 'coach', 'day_of_week', 'month', 'hour',
        'drive_time_to_attended_facility'
    ]
    if include_facility:
        feature_cols.insert(0, 'facility')

    X = df_subset[feature_cols].copy()
    y = df_subset['returned_within_30_days'].copy()

    categorical = [c for c in feature_cols if pd.api.types.is_string_dtype(X[c])]
    X = pd.get_dummies(X, columns=categorical, drop_first=False)
    X = X.astype({c: 'float' for c in X.columns if X[c].dtype == 'bool'})
    print("Features after encoding:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("Train:", len(X_train), "Test:", len(X_test))

    metrics = {'version': version_name, 'n_rows': len(df_subset), 'n_features': X.shape[1]}

    print("--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr.predict(X_test_scaled)
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_pr_auc = average_precision_score(y_test, lr_proba)
    print("ROC-AUC:", round(lr_auc, 4))
    print("PR-AUC:", round(lr_pr_auc, 4))
    print(classification_report(y_test, lr_pred, digits=3))

    coef_df = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_[0]})
    coef_df = coef_df.sort_values('coef', key=abs, ascending=False)
    print("Top 10 LR coefficients:")
    print(coef_df.head(10).to_string(index=False))

    metrics['lr_roc_auc'] = lr_auc
    metrics['lr_pr_auc'] = lr_pr_auc

    print("--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
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


fv_results = {}
fv_results['combined'] = build_first_visit_models(df, "COMBINED", include_facility=True)
fv_results['corona'] = build_first_visit_models(df[df['facility'] == 'Corona'], "CORONA ONLY", include_facility=False)
fv_results['tustin'] = build_first_visit_models(df[df['facility'] == 'Tustin'], "TUSTIN ONLY", include_facility=False)


# final summary
print("=" * 60)
print("MODEL C - GENERAL MODEL COMPARISON")
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
print("MODEL C - FIRST-VISIT SUB-MODEL COMPARISON")
print("=" * 60)

summary_fv = pd.DataFrame([
    {
        'Version': r[0]['version'],
        'Rows': r[0]['n_rows'],
        'Features': r[0]['n_features'],
        'LR ROC-AUC': round(r[0]['lr_roc_auc'], 4),
        'LR PR-AUC': round(r[0]['lr_pr_auc'], 4),
        'RF ROC-AUC': round(r[0]['rf_roc_auc'], 4),
        'RF PR-AUC': round(r[0]['rf_pr_auc'], 4),
    }
    for r in fv_results.values()
])
print(summary_fv.to_string(index=False))

print("=" * 60)
print("DONE")
print("=" * 60)