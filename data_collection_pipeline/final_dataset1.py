import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("DATASET 1: ZIP CODE LEVEL")
print("=" * 60)

d1 = pd.read_csv("dataset1_zip_level.csv")
d1["zip"] = d1["zip"].astype(str).str.zfill(5)

print(f"Loaded: {len(d1)} rows, {len(d1.columns)} columns")

#fill nulls
corona_cols = [c for c in d1.columns if c.startswith("corona_")]
tustin_cols = [c for c in d1.columns if c.startswith("tustin_")]
d1[corona_cols] = d1[corona_cols].fillna(0)
d1[tustin_cols] = d1[tustin_cols].fillna(0)
d1["median_household_income"] = d1["median_household_income"].fillna(d1["median_household_income"].median())
d1["pct_households_with_children"] = d1["pct_households_with_children"].fillna(d1["pct_households_with_children"].median())
d1["drive_time_to_corona_min"] = d1["drive_time_to_corona_min"].fillna(d1["drive_time_to_corona_min"].median())
d1["drive_time_to_tustin_min"] = d1["drive_time_to_tustin_min"].fillna(d1["drive_time_to_tustin_min"].median())
d1["drive_time_to_laguna_min"] = d1["drive_time_to_laguna_min"].fillna(d1["drive_time_to_laguna_min"].median())
d1["customers_per_1000_youth"] = d1["customers_per_1000_youth"].replace([np.inf, -np.inf], 0)
print("Filled nulls")

#drop redundant columns
d1 = d1.drop(columns=["dist_to_corona_mi", "dist_to_tustin_mi", "dist_to_laguna_mi",
                       "total_population", "households_with_children"])
print("Dropped redundant columns")

#missing values check
print()
print("Missing values:")
print(d1.isnull().sum().to_string())

#school counts by zip
print()
print("=" * 60)
print("ADDING SCHOOL COUNTS")
print("=" * 60)

schools = pd.read_csv("schools.csv", encoding="latin1", low_memory=False)
schools = schools[schools["ST"] == "CA"].copy()
schools = schools[schools["SY_STATUS_TEXT"] == "Open"].copy()
schools["zip"] = schools["LZIP"].astype(str).str.strip().str[:5]

print(f"Total California open schools: {len(schools)}")
print(f"School levels: {schools['LEVEL'].value_counts().to_string()}")

elementary = schools[schools["LEVEL"] == "Elementary"].groupby("zip").size().reset_index(name="elementary_schools")
middle = schools[schools["LEVEL"] == "Middle"].groupby("zip").size().reset_index(name="middle_schools")
high = schools[schools["LEVEL"] == "High"].groupby("zip").size().reset_index(name="high_schools")

d1 = d1.merge(elementary, on="zip", how="left")
d1 = d1.merge(middle, on="zip", how="left")
d1 = d1.merge(high, on="zip", how="left")
d1["elementary_schools"] = d1["elementary_schools"].fillna(0).astype(int)
d1["middle_schools"] = d1["middle_schools"].fillna(0).astype(int)
d1["high_schools"] = d1["high_schools"].fillna(0).astype(int)
d1["total_schools"] = d1["elementary_schools"] + d1["middle_schools"] + d1["high_schools"]

print(f"Zips with at least one school: {(d1['total_schools'] > 0).sum()} out of {len(d1)}")
print("Added: elementary_schools, middle_schools, high_schools, total_schools")

#histograms
d1_numeric = d1[["youth_pop_5_17", "median_household_income", "pct_households_with_children",
                  "drive_time_to_corona_min", "drive_time_to_tustin_min", "drive_time_to_laguna_min",
                  "total_customers", "total_visits", "customers_per_1000_youth",
                  "elementary_schools", "middle_schools", "high_schools"]].copy()

print()
print("Plotting histograms...")
n_cols = 4
n_rows = int(np.ceil(len(d1_numeric.columns) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
fig.suptitle("Dataset 1 - Zip Code Level Distributions", fontsize=13, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(d1_numeric.columns):
    axes[i].hist(d1_numeric[col].dropna(), bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

#correlation matrix
print()
print("Correlation matrix for Dataset 1:")
d1_corr = d1_numeric.corr()
print(d1_corr.round(2).to_string())
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(d1_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Dataset 1 - Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

#flag highly correlated features
print()
print("Flagging highly correlated features (|r| > 0.8):")
flagged = []
cols = d1_corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(d1_corr.iloc[i, j]) > 0.8:
            flagged.append((cols[i], cols[j], round(d1_corr.iloc[i, j], 3)))
if flagged:
    for f in flagged:
        print(f"{f[0]} <-> {f[1]}: r = {f[2]}")
else:
    print("No highly correlated features found")

#save
d1.to_csv("dataset1_final.csv", index=False)
print()
print(f"Final: {len(d1)} rows, {len(d1.columns)} columns")
print(f"Columns: {d1.columns.tolist()}")
print()
print("=" * 60)
print("Saved: dataset1_final.csv")
print("=" * 60)