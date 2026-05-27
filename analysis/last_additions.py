import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT_DIR = "goal1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DS1_PATH = "dataset1_final.csv"
DS2_PATH = "dataset2_final.csv"
DS3_PATH = "dataset3_final.csv"

NAVY   = "#0D2B4E"
BLUE   = "#1F77B4"
ACCENT = "#4FC3F7"
ORANGE = "#FF7F0E"
GREEN  = "#2CA02C"
RED    = "#D62728"
GREY   = "#6B7A99"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#DDE3EC",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlecolor": NAVY,
    "axes.labelcolor": NAVY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

ZIP_TO_CITY = {
    "90680": "Stanton",         "90805": "Long Beach",
    "92405": "San Bernardino",
    "92503": "Riverside",       "92504": "Riverside",
    "92505": "Riverside",       "92507": "Riverside",
    "92530": "Lake Elsinore",
    "92602": "Irvine",          "92603": "Irvine",
    "92604": "Irvine",          "92606": "Irvine",
    "92610": "Foothill Ranch",  "92612": "Irvine",
    "92614": "Irvine",          "92618": "Irvine",
    "92620": "Irvine",
    "92626": "Costa Mesa",      "92627": "Costa Mesa",
    "92660": "Newport Beach",
    "92629": "Dana Point",      "92630": "Lake Forest",
    "92651": "Laguna Beach",    "92653": "Laguna Hills",
    "92656": "Aliso Viejo",     "92673": "San Clemente",
    "92675": "San Juan Cap.",   "92677": "Laguna Niguel",
    "92688": "Rancho Sta. Margarita",
    "92691": "Mission Viejo",   "92692": "Mission Viejo",
    "92694": "Ladera Ranch",
    "92646": "Huntington Bch",  "92648": "Huntington Bch",
    "92649": "Huntington Bch",
    "92703": "Santa Ana",       "92704": "Santa Ana",
    "92705": "Santa Ana",       "92706": "Santa Ana",
    "92780": "Tustin",          "92782": "Tustin",
    "92806": "Anaheim",         "92807": "Anaheim",
    "92840": "Garden Grove",
    "92867": "Orange",          "92869": "Orange",
    "92870": "Placentia",
    "92879": "Corona",          "92880": "Corona",
    "92881": "Corona",          "92882": "Corona",
    "92883": "Corona",
    "92886": "Yorba Linda",
}

def zip_label(z):
    z = str(z).zfill(5)
    city = ZIP_TO_CITY.get(z)
    return f"{z} - {city}" if city else z

print("Loading datasets ...")
ds1 = pd.read_csv(DS1_PATH)
ds2 = pd.read_csv(DS2_PATH)
ds3 = pd.read_csv(DS3_PATH)

ds1["zip"] = ds1["zip"].astype(str).str.zfill(5)
ds2["zip"] = ds2["zip"].astype(str).str.zfill(5)

print(f"  DS1: {len(ds1)} ZIPs")
print(f"  DS2: {len(ds2)} customers")
print(f"  DS3: {len(ds3)} visit records")

print("\n[24] Customers per ZIP ...")
top_zip_customers = (
    ds1[ds1["total_customers"] > 0]
    .nlargest(20, "total_customers")
    [["zip", "total_customers", "corona_total_customers", "tustin_total_customers"]]
)

fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(top_zip_customers))
ax.barh(y, top_zip_customers["corona_total_customers"], color=ORANGE, label="Corona")
ax.barh(y, top_zip_customers["tustin_total_customers"],
        left=top_zip_customers["corona_total_customers"], color=BLUE, label="Tustin")
ax.set_yticks(y)
ax.set_yticklabels([zip_label(z) for z in top_zip_customers["zip"]])
ax.invert_yaxis()
ax.set_xlabel("Total Customers")
ax.set_title("Top 20 ZIP Codes by Total Customers (Stacked by Facility)")
ax.legend(loc="lower right", frameon=False)
for i, total in enumerate(top_zip_customers["total_customers"]):
    ax.text(total + 5, i, f"{int(total)}", va="center", fontsize=9, color=NAVY)
save(fig, "24_customers_per_zip.png")

print("\n[25] Avg visits per customer per ZIP ...")
ds1_eng = ds1[ds1["total_customers"] >= 10].copy()
ds1_eng["avg_visits_per_customer"] = ds1_eng["total_visits"] / ds1_eng["total_customers"]
top_eng = ds1_eng.nlargest(20, "avg_visits_per_customer")

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh([zip_label(z) for z in top_eng["zip"]],
        top_eng["avg_visits_per_customer"], color=NAVY)
ax.invert_yaxis()
ax.set_xlabel("Average Visits per Customer (lifetime)")
ax.set_title("Top 20 ZIPs by Average Visits Per Customer\n(ZIPs with >=10 customers only)")
for i, v in enumerate(top_eng["avg_visits_per_customer"]):
    ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9, color=NAVY)
save(fig, "25_avg_visits_per_zip.png")

print("\n[26] Power-user customers by ZIP ...")
power_users = ds2[ds2["avg_visits_per_month"] > 6].copy()
print(f"  Power users: {len(power_users)} (out of {len(ds2)} customers)")

power_by_zip = (
    power_users.groupby("zip")
    .size()
    .reset_index(name="power_users")
    .sort_values("power_users", ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh([zip_label(z) for z in power_by_zip["zip"]],
        power_by_zip["power_users"], color=GREEN)
ax.invert_yaxis()
ax.set_xlabel("Customers averaging >6 visits/month")
ax.set_title(f"Top 20 ZIPs Producing High-Frequency Customers (>6 visits/month)\n"
             f"Total power users across all ZIPs: {len(power_users):,}")
for i, v in enumerate(power_by_zip["power_users"]):
    ax.text(v + 0.3, i, f"{int(v)}", va="center", fontsize=9, color=NAVY)
save(fig, "26_power_users_by_zip.png")

print("\n[27] Visit volume per ZIP ...")
top_visit_zips = ds1.nlargest(20, "total_visits")

fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(top_visit_zips))
ax.barh(y, top_visit_zips["corona_total_visits"], color=ORANGE, label="Corona")
ax.barh(y, top_visit_zips["tustin_total_visits"],
        left=top_visit_zips["corona_total_visits"], color=BLUE, label="Tustin")
ax.set_yticks(y)
ax.set_yticklabels([zip_label(z) for z in top_visit_zips["zip"]])
ax.invert_yaxis()
ax.set_xlabel("Total Visits")
ax.set_title("Top 20 ZIP Codes by Total Visit Volume")
ax.legend(loc="lower right", frameon=False)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
save(fig, "27_visit_volume_per_zip.png")

print("\n[28] Active members by ZIP ...")
ds1["total_active_members"] = ds1["corona_active_members"] + ds1["tustin_active_members"]
top_member_zips = ds1.nlargest(20, "total_active_members")

fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(top_member_zips))
ax.barh(y, top_member_zips["corona_active_members"], color=ORANGE, label="Corona")
ax.barh(y, top_member_zips["tustin_active_members"],
        left=top_member_zips["corona_active_members"], color=BLUE, label="Tustin")
ax.set_yticks(y)
ax.set_yticklabels([zip_label(z) for z in top_member_zips["zip"]])
ax.invert_yaxis()
ax.set_xlabel("Active Members")
ax.set_title("Top 20 ZIP Codes by Active Paying Members")
ax.legend(loc="lower right", frameon=False)
for i, v in enumerate(top_member_zips["total_active_members"]):
    ax.text(v + 1, i, f"{int(v)}", va="center", fontsize=9, color=NAVY)
save(fig, "28_member_density_by_zip.png")

print("\n[29] Composite engagement score ...")
ds1_score = ds1[ds1["total_customers"] >= 10].copy()
ds1_score["avg_vpc"] = ds1_score["total_visits"] / ds1_score["total_customers"]
ds1_score["visit_score"] = 100 * ds1_score["total_visits"] / ds1_score["total_visits"].max()
ds1_score["depth_score"] = 100 * ds1_score["avg_vpc"]      / ds1_score["avg_vpc"].max()
ds1_score["engagement_score"] = (ds1_score["visit_score"] + ds1_score["depth_score"]) / 2

top_score = ds1_score.nlargest(20, "engagement_score")[
    ["zip", "engagement_score", "total_visits", "total_customers", "avg_vpc"]]

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh([zip_label(z) for z in top_score["zip"]],
        top_score["engagement_score"], color=NAVY)
ax.invert_yaxis()
ax.set_xlabel("Engagement Score (0-100, composite of visit volume + visits/customer)")
ax.set_title("Top 20 ZIPs by Engagement Score\n"
             "Balances reach (total visits) with depth (avg visits per customer)")
for i, row in enumerate(top_score.itertuples()):
    ax.text(row.engagement_score + 0.8, i,
            f"{row.engagement_score:.1f}  ({int(row.total_customers)} cust, {row.avg_vpc:.1f} vpc)",
            va="center", fontsize=8, color=NAVY)
ax.set_xlim(0, top_score["engagement_score"].max() * 1.6)
save(fig, "29_engagement_score_by_zip.png")

print("\n[30] Model performance comparison ...")
model_perf = pd.DataFrame({
    "Model":   ["A: Churn",  "A: Churn",   "A: Churn",
                "B: Membership", "B: Membership", "B: Membership",
                "C: Return-30d", "C: Return-30d", "C: Return-30d"],
    "Version": ["Combined", "Corona", "Tustin"] * 3,
    "RF ROC-AUC": [0.7637, 0.8052, 0.7673,
                   0.9089, 0.9473, 0.9353,
                   0.8127, 0.8149, 0.8051],
    "LR ROC-AUC": [0.7030, 0.7979, 0.7382,
                   0.9053, 0.9168, 0.9258,
                   0.7883, 0.8019, 0.7787],
})

fig, ax = plt.subplots(figsize=(11, 6))
models = model_perf["Model"].unique()
versions = model_perf["Version"].unique()
x = np.arange(len(models))
width = 0.25
colors = {"Combined": NAVY, "Corona": ORANGE, "Tustin": BLUE}

for i, v in enumerate(versions):
    sub = model_perf[model_perf["Version"] == v]
    bars = ax.bar(x + (i - 1) * width, sub["RF ROC-AUC"], width,
                  label=v, color=colors[v])
    for b, val in zip(bars, sub["RF ROC-AUC"]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=8, color=NAVY)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("ROC-AUC (Random Forest)")
ax.set_ylim(0.5, 1.0)
ax.set_title("Model Performance - Random Forest ROC-AUC Across All Models & Versions")
ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(0.7, color=GREEN, linestyle=":", linewidth=0.8, alpha=0.5)
ax.legend(loc="lower right", frameon=False, title="Facility Version")
save(fig, "30_model_performance_comparison.png")

print("\n[31] Churn feature importance ...")
churn_feats = pd.DataFrame({
    "feature": [
        "first_month", "has_second_visit", "visits_first_30d",
        "drive_time_to_facility", "pct_households_with_children",
        "youth_pop_5_17", "median_household_income",
        "returned_within_30_of_first", "returned_within_7_of_first",
        "facility_Corona", "facility_Tustin",
        "first_coach_Brandon Segura", "first_session_Camp",
        "first_coach_Mario Luna", "first_weekend",
    ],
    "importance": [0.174, 0.172, 0.092, 0.069, 0.066,
                   0.063, 0.057, 0.052, 0.027, 0.021,
                   0.021, 0.019, 0.014, 0.013, 0.012],
})

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = [RED if "month" in f or "second_visit" in f or "30d" in f else NAVY
              for f in churn_feats["feature"]]
ax.barh(churn_feats["feature"], churn_feats["importance"], color=colors_bar)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Model A - Top 15 Features Predicting Customer Churn (Combined)\n"
             "Red = strongest early-engagement signals")
for i, v in enumerate(churn_feats["importance"]):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color=NAVY)
save(fig, "31_churn_feature_importance.png")

print("\n[32] Membership feature importance ...")
mem_feats = pd.DataFrame({
    "feature": [
        "visits_first_30d", "returned_within_7_of_first",
        "returned_within_30_of_first", "has_second_visit",
        "drive_time_to_facility", "median_household_income",
        "pct_households_with_children", "youth_pop_5_17",
        "first_month", "first_coach_Touch N Go Staff",
        "first_session_Camp", "first_coach_Other",
        "first_weekend", "first_session_FT&C Beginners Youngers",
        "first_coach_Candice Silva",
    ],
    "importance": [0.379, 0.151, 0.094, 0.084, 0.057,
                   0.039, 0.037, 0.036, 0.029, 0.008,
                   0.007, 0.007, 0.007, 0.007, 0.006],
})

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = [GREEN if "30d" in f or "7_of_first" in f or "second_visit" in f else NAVY
              for f in mem_feats["feature"]]
ax.barh(mem_feats["feature"], mem_feats["importance"], color=colors_bar)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Model B - Top 15 Features Predicting Membership Conversion (Combined)\n"
             "Green = early-engagement signals (dominant predictors)")
for i, v in enumerate(mem_feats["importance"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=NAVY)
save(fig, "32_membership_feature_importance.png")

print("\n[33] Huff cannibalization scenarios ...")
huff = pd.DataFrame({
    "Scenario":            ["Laguna = Corona\n(low attractiveness)",
                            "Laguna = Midpoint\n(geometric mean)",
                            "Laguna = Tustin\n(high attractiveness)"],
    "Corona Visits Lost":  [0,    324,  515],
    "Tustin Visits Lost":  [1519, 2891, 6369],
    "Total Cannibalized":  [1519, 3215, 6884],
})

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(huff))
width = 0.35
b1 = ax.bar(x - width / 2, huff["Corona Visits Lost"], width, color=ORANGE, label="Corona Visits Lost")
b2 = ax.bar(x + width / 2, huff["Tustin Visits Lost"], width, color=BLUE,   label="Tustin Visits Lost")
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h + 60, f"{int(h):,}",
                    ha="center", fontsize=9, color=NAVY)
ax.set_xticks(x)
ax.set_xticklabels(huff["Scenario"])
ax.set_ylabel("Annual Visits Cannibalized (estimated)")
ax.set_title("Model D - Huff Spatial Choice: Cannibalization by Laguna Attractiveness Scenario\n"
             "Beta (distance decay) = 2.0")
ax.legend(loc="upper left", frameon=False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
save(fig, "33_huff_cannibalization.png")

print("\n[34] Top ZIPs forecasted for Laguna ...")
laguna_top = pd.DataFrame({
    "zip": ["92677", "92656", "92629", "92692", "92653",
            "92675", "92691", "92694", "92688", "92651",
            "92630", "92673", "92660", "92618", "92880"],
    "laguna_customers_rf": [84.9, 65.5, 40.6, 28.5, 28.2,
                            26.7, 23.9, 19.5, 19.2, 17.8,
                            17.6, 16.8, 10.8, 10.6, 9.2],
    "drive_time_min":      [1.4, 9.2, 8.1, 15.7, 10.1,
                            13.6, 16.0, 20.0, 17.9, 13.1,
                            18.2, 22.8, 23.1, 23.4, 54.8],
    "cannibalized":        [43.7, 37.9, 6.0, 24.3, 12.4,
                            11.4, 18.0, 28.9, 15.7, 5.1,
                            0.0, 6.3, 0.0, 0.0, 0.0],
})
laguna_top["net_new"] = laguna_top["laguna_customers_rf"] - laguna_top["cannibalized"]

fig, ax = plt.subplots(figsize=(13, 7))
y = np.arange(len(laguna_top))
ax.barh(y, laguna_top["cannibalized"], color=RED, label="Cannibalized (from Corona/Tustin)")
ax.barh(y, laguna_top["net_new"], left=laguna_top["cannibalized"],
        color=GREEN, label="Net New Customers")
ax.set_yticks(y)
ax.set_yticklabels([f"{zip_label(z)} ({d:.0f} min)" for z, d in
                    zip(laguna_top["zip"], laguna_top["drive_time_min"])])
ax.invert_yaxis()
ax.set_xlabel("Predicted Laguna Customers (Random Forest)")
ax.set_title("Model E - Top 15 ZIPs Forecasted to Feed a Laguna Location\n"
             "Split between net new demand (green) vs cannibalized (red)\n"
             "Total RF forecast: 791 customers - 559 net new - 232 cannibalized")
ax.legend(loc="lower right", frameon=False)
for i, row in enumerate(laguna_top.itertuples()):
    ax.text(row.laguna_customers_rf + 1.5, i, f"{row.laguna_customers_rf:.1f}",
            va="center", fontsize=9, color=NAVY)
save(fig, "34_laguna_top_zips_forecast.png")

print("\n[35] Beta sensitivity ...")
beta_sens = pd.DataFrame({
    "beta":            [1.0, 1.5, 2.0, 2.5, 3.0],
    "mean_p_laguna":   [0.276, 0.278, 0.281, 0.285, 0.289],
    "total_cannibalized": [2630, 2951, 3215, 3451, 3664],
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.plot(beta_sens["beta"], beta_sens["mean_p_laguna"], marker="o",
         color=NAVY, linewidth=2, markersize=8)
ax1.set_xlabel("Beta (distance decay exponent)")
ax1.set_ylabel("Mean P(Laguna) - Near ZIPs (<30 min)")
ax1.set_title("Probability of choosing Laguna vs Beta")
ax1.grid(alpha=0.3)
for x_val, y_val in zip(beta_sens["beta"], beta_sens["mean_p_laguna"]):
    ax1.annotate(f"{y_val:.3f}", (x_val, y_val), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9, color=NAVY)

ax2 = axes[1]
bars = ax2.bar(beta_sens["beta"].astype(str), beta_sens["total_cannibalized"],
               color=RED, alpha=0.85)
ax2.set_xlabel("Beta (distance decay exponent)")
ax2.set_ylabel("Total Cannibalized Visits (annual)")
ax2.set_title("Cannibalization Volume vs Beta (Midpoint Scenario)")
for b, val in zip(bars, beta_sens["total_cannibalized"]):
    ax2.text(b.get_x() + b.get_width() / 2, val + 50, f"{int(val):,}",
             ha="center", fontsize=9, color=NAVY)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))

fig.suptitle("Model D - Huff Beta Sensitivity Analysis", fontsize=14,
             fontweight="bold", color=NAVY, y=1.02)
save(fig, "35_huff_beta_sensitivity.png")

print("\nAll charts generated.")
print(f"Charts directory: {os.path.abspath(OUT_DIR)}")