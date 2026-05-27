import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("DATASET 3: VISIT LEVEL")
print("=" * 60)

d3 = pd.read_csv("dataset3_visit_level.csv", low_memory=False)
d3["date"] = pd.to_datetime(d3["date"], errors="coerce")
d3["client_id"] = d3["client_id"].astype(str)

print(f"Loaded: {len(d3)} rows, {len(d3.columns)} columns")

#drop unnecessary columns
d3 = d3.drop(columns=["created_by"])
print("Dropped: created_by")

#remove duplicates
before = len(d3)
d3 = d3.drop_duplicates()
print(f"Removed {before - len(d3)} duplicate rows")

#missing values
print()
print("Missing values:")
print(d3.isnull().sum().to_string())

#add returned_within_30_days flag
print()
print("Calculating returned_within_30_days...")
d3_sorted = d3.sort_values(["client_id", "date"]).copy()
d3_sorted["next_visit_date"] = d3_sorted.groupby("client_id")["date"].shift(-1)
d3_sorted["days_to_next_visit"] = (d3_sorted["next_visit_date"] - d3_sorted["date"]).dt.days
d3_sorted["returned_within_30_days"] = (d3_sorted["days_to_next_visit"] <= 30).astype(int)
d3 = d3_sorted.drop(columns=["next_visit_date", "days_to_next_visit"])
print("Added: returned_within_30_days")
print(f"Visits where customer returned within 30 days: {d3['returned_within_30_days'].sum()} ({d3['returned_within_30_days'].mean()*100:.1f}%)")

#year, month, hour bar charts
print()
print("Plotting year, month, hour distributions...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Dataset 3 - Year, Month, Hour Distributions", fontsize=13, fontweight="bold")

year_counts = d3["year"].value_counts().sort_index()
axes[0].bar(year_counts.index.astype(str), year_counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
axes[0].set_title("Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Count")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

month_counts = d3["month"].value_counts().sort_index()
axes[1].bar(month_counts.index.astype(str), month_counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
axes[1].set_title("Month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Count")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

hour_counts = d3["hour"].value_counts().sort_index()
axes[2].bar(hour_counts.index.astype(str), hour_counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
axes[2].set_title("Hour")
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("Count")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()

#session category bar chart
print("Plotting session category breakdown...")
session_counts = d3["session_category"].value_counts()
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(session_counts)), session_counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(session_counts)))
ax.set_xticklabels(session_counts.index, rotation=40, ha="right", fontsize=9)
ax.set_title("Dataset 3 - Visits by Session Category", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Visits")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#day of week bar chart
print("Plotting day of week breakdown...")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = d3["day_of_week"].value_counts().reindex(day_order)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(day_order, day_counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_title("Dataset 3 - Visits by Day of Week", fontsize=13, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Number of Visits")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#facility bar chart
print("Plotting facility breakdown...")
facility_counts = d3["facility"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(facility_counts.index, facility_counts.values, color=["steelblue", "coral"], edgecolor="white", linewidth=0.5)
ax.set_title("Dataset 3 - Visits by Facility", fontsize=13, fontweight="bold")
ax.set_xlabel("Facility")
ax.set_ylabel("Number of Visits")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#am/pm bar chart
print("Plotting AM/PM breakdown...")
ampm_counts = d3["am_pm"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(ampm_counts.index, ampm_counts.values, color=["steelblue", "coral"], edgecolor="white", linewidth=0.5)
ax.set_title("Dataset 3 - AM vs PM Visits", fontsize=13, fontweight="bold")
ax.set_xlabel("Time of Day")
ax.set_ylabel("Number of Visits")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#correlation matrix
print()
print("Correlation matrix for Dataset 3:")
d3_corr = d3[["year", "month", "hour"]].corr()
print(d3_corr.round(2).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(d3_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Dataset 3 - Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

#flag highly correlated features
print()
print("Flagging highly correlated features (|r| > 0.8):")
flagged = []
cols = d3_corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(d3_corr.iloc[i, j]) > 0.8:
            flagged.append((cols[i], cols[j], round(d3_corr.iloc[i, j], 3)))
if flagged:
    for f in flagged:
        print(f"{f[0]} <-> {f[1]}: r = {f[2]}")
else:
    print("No highly correlated features found")

#save
d3.to_csv("dataset3_final.csv", index=False)
print()
print(f"Final: {len(d3)} rows, {len(d3.columns)} columns")
print(f"Columns: {d3.columns.tolist()}")
print()
print("=" * 60)
print("Saved: dataset3_final.csv")
print("=" * 60)