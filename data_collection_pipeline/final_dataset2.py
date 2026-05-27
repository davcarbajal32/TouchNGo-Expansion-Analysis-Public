import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("DATASET 2: CUSTOMER LEVEL")
print("=" * 60)

d2 = pd.read_csv("dataset2_customer_level.csv")
d3 = pd.read_csv("dataset3_visit_level.csv", low_memory=False)
distances = pd.read_csv("socal_zip_distances.csv")

d2["client_id"] = d2["client_id"].astype(str)
d2["zip"] = d2["zip"].astype(str).str.strip().str.zfill(5)
d3["client_id"] = d3["client_id"].astype(str)
d3["date"] = pd.to_datetime(d3["date"], errors="coerce")

#dedup distances so each zip only appears once
distances["zip"] = distances["zip"].astype(str).str.zfill(5)
distances = distances.drop_duplicates(subset=["zip"]).reset_index(drop=True)

print(f"Loaded: {len(d2)} rows, {len(d2.columns)} columns")

#drop columns not useful for modeling
d2 = d2.drop(columns=["first_visit", "last_visit", "membership_joined_on",
                       "membership_status", "membership_tier", "City", "State", "lifetime_sales"])
print("Dropped non-feature columns")

#missing values
print()
print("Missing values:")
print(d2.isnull().sum().to_string())

#add drive times
print(f"\nBefore distances merge: {len(d2)} rows")
d2 = d2.merge(distances[["zip", "drive_time_to_corona_min", "drive_time_to_tustin_min", "drive_time_to_laguna_min"]], on="zip", how="left")
print(f"After distances merge: {len(d2)} rows")
d2["drive_time_to_corona_min"] = d2["drive_time_to_corona_min"].fillna(d2["drive_time_to_corona_min"].median())
d2["drive_time_to_tustin_min"] = d2["drive_time_to_tustin_min"].fillna(d2["drive_time_to_tustin_min"].median())
d2["drive_time_to_laguna_min"] = d2["drive_time_to_laguna_min"].fillna(d2["drive_time_to_laguna_min"].median())
print("Added: drive_time_to_corona_min, drive_time_to_tustin_min, drive_time_to_laguna_min")

#add drive time to the specific facility the customer attends
def get_facility_drive_time(row):
    if row["facility"] == "Corona":
        return row["drive_time_to_corona_min"]
    elif row["facility"] == "Tustin":
        return row["drive_time_to_tustin_min"]
    return None

d2["drive_time_to_attended_facility"] = d2.apply(get_facility_drive_time, axis=1)
d2["drive_time_to_attended_facility"] = d2["drive_time_to_attended_facility"].fillna(d2["drive_time_to_attended_facility"].median())
print("Added: drive_time_to_attended_facility")

#add months since first visit
d2_original = pd.read_csv("dataset2_customer_level.csv")
d2_original["client_id"] = d2_original["client_id"].astype(str)
d2_original["facility"] = d2_original["facility"].astype(str)
d2_original["first_visit"] = pd.to_datetime(d2_original["first_visit"], errors="coerce")
reference_date = pd.Timestamp("2026-04-13")
d2_original["months_since_first_visit"] = ((reference_date - d2_original["first_visit"]).dt.days / 30).round(1)
d2_original = d2_original.drop_duplicates(subset=["client_id", "facility"]).reset_index(drop=True)

print(f"\nBefore months merge: {len(d2)} rows")
d2 = d2.merge(d2_original[["client_id", "facility", "months_since_first_visit"]], on=["client_id", "facility"], how="left")
print(f"After months merge: {len(d2)} rows")
d2["months_since_first_visit"] = d2["months_since_first_visit"].fillna(0)
print("Added: months_since_first_visit")

#add visit streak
print("Calculating visit streaks...")

d3["facility_client"] = d3["facility"] + "_" + d3["client_id"]
visit_months = d3.groupby("facility_client").apply(
    lambda x: set(zip(x["year"].astype(int), x["month"].astype(int)))
).reset_index()
visit_months.columns = ["facility_client", "visit_month_set"]

def calc_streak(month_set):
    if not month_set:
        return 0
    sorted_months = sorted(month_set)
    max_streak = 1
    current_streak = 1
    for i in range(1, len(sorted_months)):
        y1, m1 = sorted_months[i-1]
        y2, m2 = sorted_months[i]
        diff = (y2 - y1) * 12 + (m2 - m1)
        if diff == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    return max_streak

visit_months["visit_streak"] = visit_months["visit_month_set"].apply(calc_streak)
d2["facility_client"] = d2["facility"] + "_" + d2["client_id"]

print(f"\nBefore visit_streak merge: {len(d2)} rows")
d2 = d2.merge(visit_months[["facility_client", "visit_streak"]], on="facility_client", how="left")
print(f"After visit_streak merge: {len(d2)} rows")
d2 = d2.drop(columns=["facility_client"])
d2["visit_streak"] = d2["visit_streak"].fillna(0).astype(int)
print("Added: visit_streak")

print(f"\nFinal row count: {len(d2)} rows")

#histograms
d2_numeric = d2[["total_visits", "days_active", "days_since_last_visit",
                  "avg_visits_per_month", "unique_session_types", "unique_coaches",
                  "drive_time_to_attended_facility", "months_since_first_visit",
                  "visit_streak"]].copy()

print()
print("Plotting histograms...")
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("Dataset 2 - Customer Level Distributions", fontsize=13, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(d2_numeric.columns):
    axes[i].hist(d2_numeric[col].dropna(), bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

#facility bar chart
print("Plotting facility breakdown...")
fac_counts = d2["facility"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(fac_counts.index, fac_counts.values, color=["steelblue", "coral"], edgecolor="white", linewidth=0.5)
ax.set_title("Dataset 2 - Customers by Facility", fontsize=13, fontweight="bold")
ax.set_xlabel("Facility")
ax.set_ylabel("Number of Customers")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#churn bar chart
print("Plotting churn breakdown...")
churn_counts = d2["is_churned"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Retained", "Churned"], churn_counts.values, color=["steelblue", "coral"], edgecolor="white", linewidth=0.5)
ax.set_title("Dataset 2 - Churned vs Retained Customers", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#correlation matrix
print()
print("Correlation matrix for Dataset 2:")
d2_corr = d2_numeric.corr()
print(d2_corr.round(2).to_string())
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(d2_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Dataset 2 - Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

#flag highly correlated features
print()
print("Flagging highly correlated features (|r| > 0.8):")
flagged = []
cols = d2_corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(d2_corr.iloc[i, j]) > 0.8:
            flagged.append((cols[i], cols[j], round(d2_corr.iloc[i, j], 3)))
if flagged:
    for f in flagged:
        print(f"{f[0]} <-> {f[1]}: r = {f[2]}")
else:
    print("No highly correlated features found")

d2 = d2.drop(columns=["visit_streak"])
print("Dropped visit_streak (r=0.843 with total_visits)")
d2 = d2.drop_duplicates().reset_index(drop=True)
print(f"After removing fully duplicate rows: {len(d2)} rows")
print(f"Corona customers: {len(d2[d2['facility'] == 'Corona'])}")
print(f"Tustin customers: {len(d2[d2['facility'] == 'Tustin'])}")
#save

d2.to_csv("dataset2_final.csv", index=False)
print()
print(f"Final: {len(d2)} rows, {len(d2.columns)} columns")
print(f"Columns: {d2.columns.tolist()}")
print()
print("=" * 60)
print("Saved: dataset2_final.csv")
print("=" * 60)