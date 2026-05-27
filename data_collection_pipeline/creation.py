import pandas as pd
import numpy as np


#load all source files
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)


corona_clients = pd.read_csv("coronamailinglist.csv", encoding="latin1")
tustin_clients = pd.read_csv("tustinmailinglist.csv", encoding="latin1")
corona_visits = pd.read_csv("corona_all_visits_cleaned.csv")
tustin_visits = pd.read_csv("tustin_all_visits_cleaned.csv")
corona_memberships = pd.read_csv("corona_memberships.csv", encoding="latin1")
tustin_memberships = pd.read_csv("tustin_memberships.csv", encoding="latin1")
census = pd.read_csv("socal_census_data.csv")
distances = pd.read_csv("socal_zip_distances.csv")

#add location column to visits
corona_visits["facility"] = "Corona"
tustin_visits["facility"] = "Tustin"

#standardize client id column in mailing lists
corona_clients["client_id"] = corona_clients["ID"].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)
tustin_clients["client_id"] = tustin_clients["ID"].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)

#standardize client id in memberships
corona_memberships["client_id"] = corona_memberships["BarcodeID"].astype(str)
tustin_memberships["client_id"] = tustin_memberships["BarcodeID"].astype(str)

#standardize zip codes in mailing lists
corona_clients["zip"] = corona_clients["Postal code"].astype(str).str.strip().str[:5]
tustin_clients["zip"] = tustin_clients["Postal code"].astype(str).str.strip().str[:5]

print(f"Corona clients: {len(corona_clients)}")
print(f"Tustin clients: {len(tustin_clients)}")
print(f"Corona visits: {len(corona_visits)}")
print(f"Tustin visits: {len(tustin_visits)}")
print(f"Corona memberships: {len(corona_memberships)}")
print(f"Tustin memberships: {len(tustin_memberships)}")


print()
print("=" * 60)
print("BUILDING DATASET 3: Visit Level Dataset")
print("=" * 60)

#combine both locations visit history
all_visits = pd.concat([corona_visits, tustin_visits], ignore_index=True)

all_visits["coach"] = all_visits["coach"].fillna(all_visits["teacher"])
all_visits = all_visits.drop(columns=["teacher"])

all_visits["date"] = pd.to_datetime(all_visits["date"], errors="coerce")

all_visits["date"] = pd.to_datetime(all_visits["date"], errors="coerce")
all_visits["year"] = all_visits["date"].dt.year
all_visits["month"] = all_visits["date"].dt.month
all_visits["month_name"] = all_visits["date"].dt.strftime("%B")
all_visits["day_of_week"] = all_visits["date"].dt.day_name()

"""import pandas as pd
visits = pd.read_csv("dataset3_visit_level.csv")
session_counts = visits["description"].value_counts()
print(session_counts.to_string())"""


all_visits["hour"] = all_visits["time"].str.extract(r'(\d+):').astype(float)
all_visits["am_pm"] = all_visits["time"].str.extract(r'(AM|PM)')
#session type mapping
def categorize_session(desc):
    if pd.isna(desc):
        return None
    d = desc.upper()

    # Camps
    camp_keywords = ["CAMP", "HELL WEEK", "WINTER CAMP", "SPRING BREAK", "SUMMER",
                     "PRESIDENT", "MEMORIAL", "LABOR DAY", "VETERAN", "THANKSGIVING",
                     "WORLD CUP", "TECHNICAL CAMP", "ACADEMY CAMP", "MASTERCLASS",
                     "MASTERY CAMP", "FINISHING FROM ANYWHERE", "READING THE GAME",
                     "PASSING & COMMUNICATION", "MIDFIELD COMMAND", "MASTERING DEFENSE",
                     "OFFENSE & DEFENSE COLLIDE", "TACTICAL MASTERY"]
    if any(k in d for k in camp_keywords):
        return "Camp"

    # Juniors (no minis)
    if "JUNIOR" in d and "MINIS" not in d and "MINI" not in d:
        return "Juniors"

    # Privates
    if any(k in d for k in ["PRIVATE", "PRIVATES"]):
        return "Private Sessions"

    # Defense
    if any(k in d for k in ["DEFENSE", "DEFENSIVE", "PHYSICALITY"]):
        if "OLDER" in d:
            return "Defense - Olders"
        return "Defense - Youngers"

    # Ball Mastery
    if "BALL MASTERY" in d:
        if "INTERMEDIATE" in d:
            return "Ball Mastery - Intermediate - Youngers"
        if "BEGINNER" in d and "YOUNGER" in d:
            return "Ball Mastery - Beginners - Youngers"
        return "Ball Mastery - Beginners - Youngers"

    # Shooting & Finishing
    if any(k in d for k in ["SHOOTING", "FINISHING"]):
        if "OLDER" in d:
            if any(k in d for k in ["INTERMEDIATE", "ADVANCE", "ADVANCED"]):
                return "Shooting & Finishing - Intermediate / Advanced - Olders"
            return "Shooting & Finishing - Beginners - Olders"
        if "INTERMEDIATE" in d:
            return "Shooting & Finishing - Intermediate - Youngers"
        return "Shooting & Finishing - Beginners - Youngers"

    # First Touch & Control
    if "FIRST TOUCH" in d:
        if "OLDER" in d:
            return "First Touch & Control - Beginners - Olders"
        return "First Touch & Control - Beginners - Youngers"

    # Speed of Play
    if "SPEED OF PLAY" in d:
        if "OLDER" in d:
            return "Speed of Play - Advanced - Olders"
        return "Speed of Play - Advanced - Youngers"

    # Passing & Movement
    if "PASSING" in d:
        if "OLDER" in d:
            return "Passing & Movement - Intermediate - Olders"
        return "Passing & Movement - Intermediate - Youngers"

    # Strength & Conditioning
    if any(k in d for k in ["STRENGTH", "CONDITIONING"]):
        return "Strength & Conditioning"

    # Legacy classes - only say Beginners Youngers/Olders
    if "BEGINNER" in d and "YOUNGER" in d:
        return "First Touch & Control - Beginners - Youngers"
    if "BEGINNER" in d and "OLDER" in d:
        return "First Touch & Control - Beginners - Olders"

    # Legacy classes - only say Intermediate Youngers/Olders
    if "INTERMEDIATE" in d and "YOUNGER" in d:
        return "Passing & Movement - Intermediate - Youngers"
    if "INTERMEDIATE" in d and "OLDER" in d:
        return "Passing & Movement - Intermediate - Olders"

    # Legacy classes - only say Advanced Youngers/Olders
    if any(k in d for k in ["ADVANCE", "ADVANCED"]) and "YOUNGER" in d:
        return "Speed of Play - Advanced - Youngers"
    if any(k in d for k in ["ADVANCE", "ADVANCED"]) and "OLDER" in d:
        return "Speed of Play - Advanced - Olders"

    return None
all_visits["session_category"] = all_visits["description"].apply(categorize_session)
all_visits = all_visits[all_visits["session_category"].notna()].reset_index(drop=True)
print("\nSession category breakdown:")
print(all_visits["session_category"].value_counts(dropna=False).to_string())
all_visits = all_visits.drop(columns=["resource(s)"])
all_visits["client_id"] = all_visits["client_id"].astype(str)
all_visits.to_csv("dataset3_visit_level.csv", index=False)

print(f"Total visit records: {len(all_visits)}")
print(f"Corona visits: {len(all_visits[all_visits['facility'] == 'Corona'])}")
print(f"Tustin visits: {len(all_visits[all_visits['facility'] == 'Tustin'])}")
print(f"Date range: {all_visits['date'].min().date()} to {all_visits['date'].max().date()}")
print(f"Unique clients: {all_visits['client_id'].nunique()}")
print(f"Unique session types: {all_visits['description'].nunique()}")
print(f"Unique coaches: {all_visits['coach'].nunique()}")
print("Saved: dataset3_visit_level.csv")



print()
print("=" * 60)
print("BUILDING DATASET 2: Customer Level Dataset")
print("=" * 60)

#visit history per customer
visit_features = all_visits.groupby(["client_id", "facility"]).agg(
    total_visits=("date", "count"),
    first_visit=("date", "min"),
    last_visit=("date", "max"),
    most_common_session=("description", lambda x: x.mode()[0] if len(x.mode()) > 0 else None),
    most_common_coach=("coach", lambda x: x.mode()[0] if len(x.mode()) > 0 else None),
    unique_session_types=("description", "nunique"),
    unique_coaches=("coach", "nunique"),
).reset_index()

visit_features["days_active"] = (visit_features["last_visit"] - visit_features["first_visit"]).dt.days
visit_features["days_since_last_visit"] = (pd.Timestamp.today() - visit_features["last_visit"]).dt.days
visit_features["avg_visits_per_month"] = (visit_features["total_visits"] / ((visit_features["days_active"] / 30) + 1)).round(2)

#combine corona and tustin clients
corona_clients["facility"] = "Corona"
tustin_clients["facility"] = "Tustin"
all_clients = pd.concat([corona_clients, tustin_clients], ignore_index=True)
all_clients["client_id"] = all_clients["client_id"].astype(str)

#combine corona and tustin memberships
all_memberships = pd.concat([corona_memberships, tustin_memberships], ignore_index=True)
all_memberships["client_id"] = all_memberships["client_id"].astype(str)

#keep only relevant membership columns
membership_cols = ["client_id", "Status", "Membership Tier", "Joined On", "Lifetime Sales"]
corona_memberships_clean = corona_memberships[["client_id", "Status", "Membership Tier", "Joined On", "Lifetime Sales"]].copy()
tustin_memberships_clean = tustin_memberships[["client_id", "Status", "Membership Tier", "Joined On"]].copy()
tustin_memberships_clean["Lifetime Sales"] = None
all_memberships_clean = pd.concat([corona_memberships_clean, tustin_memberships_clean], ignore_index=True)
all_memberships_clean.columns = ["client_id", "membership_status", "membership_tier", "membership_joined_on", "lifetime_sales"]

#merge clients + visits + memberships
customer_dataset = all_clients[["client_id", "zip", "facility", "City", "State"]].copy()
customer_dataset = customer_dataset.merge(visit_features, on=["client_id", "facility"], how="left")
customer_dataset = customer_dataset.merge(all_memberships_clean, on="client_id", how="left")
customer_dataset = customer_dataset[customer_dataset["total_visits"].notna()].reset_index(drop=True)


#churn flag: no visit in last 90 days
customer_dataset["is_churned"] = (customer_dataset["days_since_last_visit"] > 90).astype(int)

#member flag
customer_dataset["is_active_member"] = (customer_dataset["membership_status"] == "Active").astype(int)

#clean lifetime sales
customer_dataset["lifetime_sales"] = pd.to_numeric(
    customer_dataset["lifetime_sales"].astype(str).str.replace("$", "").str.replace(",", ""),
    errors="coerce"
)

customer_dataset.to_csv("dataset2_customer_level.csv", index=False)

print(f"Total customers: {len(customer_dataset)}")
print(f"Customers with visits: {customer_dataset['total_visits'].notna().sum()}")
print(f"Active members: {customer_dataset['is_active_member'].sum()}")
print(f"Churned customers: {customer_dataset['is_churned'].sum()}")
print("Saved: dataset2_customer_level.csv")



print()
print("=" * 60)
print("BUILDING DATASET 1: Zip Code Level Dataset")
print("=" * 60)

#filter to valid SoCal zips
valid_zips = census["zip"].astype(str).tolist()

#aggregate customer data by zip for each location
def zip_agg(clients_df, visits_df, memberships_df, location_name):
    clients_df = clients_df.copy()
    clients_df["client_id"] = clients_df["client_id"].astype(str)
    clients_df["zip"] = clients_df["Postal code"].astype(str).str.strip().str[:5]

    #visits per client
    visit_counts = visits_df.groupby("client_id").agg(
        total_visits=("date", "count"),
        first_visit=("date", "min"),
        last_visit=("date", "max"),
    ).reset_index()
    visit_counts["client_id"] = visit_counts["client_id"].astype(str)
    visit_counts["last_visit"] = pd.to_datetime(visit_counts["last_visit"], errors="coerce")
    visit_counts["days_since_last_visit"] = (pd.Timestamp.today() - visit_counts["last_visit"]).dt.days

    #merge clients with visits
    merged = clients_df.merge(visit_counts, on="client_id", how="left")
    merged["is_churned"] = (merged["days_since_last_visit"] > 90).astype(float)
    merged["has_visits"] = merged["total_visits"].notna().astype(int)

    #membership info
    memberships_df = memberships_df.copy()
    memberships_df["client_id"] = memberships_df["client_id"].astype(str)
    memberships_df["is_active_member"] = (memberships_df["Status"] == "Active").astype(int)
    merged = merged.merge(memberships_df[["client_id", "is_active_member"]], on="client_id", how="left")

    #aggregate by zip
    zip_stats = merged.groupby("zip").agg(
        total_customers=(("client_id"), "count"),
        customers_with_visits=(("has_visits"), "sum"),
        total_visits=(("total_visits"), "sum"),
        avg_visits_per_customer=(("total_visits"), "mean"),
        active_members=(("is_active_member"), "sum"),
        churned_customers=(("is_churned"), "sum"),
    ).reset_index()

    zip_stats["retention_rate"] = (
        (zip_stats["customers_with_visits"] - zip_stats["churned_customers"]) /
        zip_stats["customers_with_visits"].replace(0, np.nan)
    ).round(3)

    zip_stats["membership_conversion_rate"] = (
        zip_stats["active_members"] / zip_stats["total_customers"]
    ).round(3)

    zip_stats.columns = ["zip"] + [f"{location_name.lower()}_{c}" for c in zip_stats.columns[1:]]
    return zip_stats

corona_zip = zip_agg(corona_clients, corona_visits, corona_memberships, "corona")
tustin_zip = zip_agg(tustin_clients, tustin_visits, tustin_memberships, "tustin")

#build zip code master
zip_master = census.copy()
zip_master["zip"] = zip_master["zip"].astype(str)
distances["zip"] = distances["zip"].astype(str).str.zfill(5)
zip_master = zip_master.merge(distances[["zip", "dist_to_corona_mi", "dist_to_tustin_mi", "dist_to_laguna_mi",
                                         "drive_time_to_corona_min", "drive_time_to_tustin_min", "drive_time_to_laguna_min"]],
                               on="zip", how="left")
zip_master = zip_master.merge(corona_zip, on="zip", how="left")
zip_master = zip_master.merge(tustin_zip, on="zip", how="left")

#combined totals
zip_master["total_customers"] = zip_master["corona_total_customers"].fillna(0) + zip_master["tustin_total_customers"].fillna(0)
zip_master["total_visits"] = zip_master["corona_total_visits"].fillna(0) + zip_master["tustin_total_visits"].fillna(0)
zip_master["customers_per_1000_youth"] = (zip_master["total_customers"] / zip_master["youth_pop_5_17"] * 1000).round(2)
zip_master["customers_per_1000_youth"] = zip_master[("customers_per_1"
                                                     "000_youth")].replace([np.inf, -np.inf], 0)
zip_master.to_csv("dataset1_zip_level.csv", index=False)

print(f"Total zip codes: {len(zip_master)}")
print(f"Zips with TG customers: {(zip_master['total_customers'] > 0).sum()}")
print(f"Total customers mapped: {zip_master['total_customers'].sum():.0f}")
print(f"Total visits mapped: {zip_master['total_visits'].sum():.0f}")
print("Saved: dataset1_zip_level.csv")

print()
print("=" * 60)
print("ALL DATASETS COMPLETE")
print("Output files:")
print("  - dataset1_zip_level.csv")
print("  - dataset2_customer_level.csv")
print("  - dataset3_visit_level.csv")
print("=" * 60)

no_zip = customer_dataset[customer_dataset["zip"].isna() | (customer_dataset["zip"] == "nan")]
print(f"\nCustomers with no zip code: {len(no_zip)}")
print(f"Total visits from these customers: {no_zip['total_visits'].sum():.0f}")
print(f"Avg visits per customer: {no_zip['total_visits'].mean():.1f}")
print(f"Active members among them: {no_zip['is_active_member'].sum()}")
print(f"Facility breakdown:")
print(no_zip["facility"].value_counts().to_string())
print(no_zip[["client_id", "facility", "total_visits", "most_common_session", "most_common_coach"]].head(20).to_string())