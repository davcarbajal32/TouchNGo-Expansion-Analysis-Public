import requests
import pandas as pd
from geopy.distance import geodesic
import time

CENSUS_API_KEY = "7ec27f2053ac48a0cc70d4e371cc58089809242c"
CORONA_ADDRESS = "280 Teller St #130, Corona, CA 92879"
TUSTIN_ADDRESS = "15042 Parkway Loop Ste C, Tustin, CA 92780"
LAGUNA_ADDRESS = "Laguna Niguel, CA"  # City center as placeholder

CORONA_COORDS = (33.8617, -117.5748)
TUSTIN_COORDS = (33.7297, -117.7946)
LAGUNA_COORDS = (33.5225, -117.7075)

#data cleaning
print("=" * 60)
print("STEP 1: Loading mailing lists")
print("=" * 60)

corona_clients = pd.read_csv("coronamailinglist.csv", encoding="latin1")
tustin_clients = pd.read_csv("tustinmailinglist.csv", encoding="latin1")
print(f"Corona clients: {len(corona_clients)}")
print(f"Tustin clients: {len(tustin_clients)}")


#removing pickup soccer games from visit history
print()
print("=" * 60)
print("STEP 2: Cleaning visit history")
print("=" * 60)

corona_visits_raw = pd.read_csv("corona_all_visits.csv", encoding="latin1")
corona_visits = corona_visits_raw[corona_visits_raw["description"] != "Indoor Pick Up Games"]
corona_visits.to_csv("corona_all_visits_cleaned.csv", index=False)
print(f"Corona visit data before: {len(corona_visits_raw)}")
print(f"Corona visit data after: {len(corona_visits)}")

tustin_visits_raw = pd.read_csv("tustin_all_visits.csv", encoding="latin1")
tustin_visits = tustin_visits_raw[tustin_visits_raw["description"] != "Indoor Pick Up Games"]
tustin_visits.to_csv("tustin_all_visits_cleaned.csv", index=False)
print(f"Tustin visit data before: {len(tustin_visits_raw)}")
print(f"Tustin visit data after: {len(tustin_visits)}")


#extracting zip codes
print()
print("=" * 60)
print("STEP 3: Extracting SoCal zip codes")
print("=" * 60)

all_zip_codes = pd.concat([
    corona_clients["Postal code"],
    tustin_clients["Postal code"]
]).dropna().astype(str).str.strip().str[:5]

valid_zips = all_zip_codes[all_zip_codes.str.match(r'^\d{5}$')]
unique_zips = sorted(valid_zips.unique())
socal_zips = [z for z in unique_zips if 90000 <= int(z) <= 93999]

print(f"Total valid zip codes: {len(unique_zips)}")
print(f"SoCal zip codes: {len(socal_zips)}")

#gathering geo data
print()
print("=" * 60)
print("STEP 4: Pulling Census ACS 5-Year (2022) data")
print("=" * 60)

BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

VARIABLES = [
    "B01003_001E",  # Total population
    "B01001_004E",  # Male 5-9
    "B01001_005E",  # Male 10-14
    "B01001_006E",  # Male 15-17
    "B01001_028E",  # Female 5-9
    "B01001_029E",  # Female 10-14
    "B01001_030E",  # Female 15-17
    "B19013_001E",  # Median household income
    "B11005_001E",  # Total households
    "B11005_002E",  # Households with children under 18
]

var_string = ",".join(VARIABLES)
url = f"{BASE_URL}?get=NAME,{var_string}&for=zip%20code%20tabulation%20area:*&key={CENSUS_API_KEY}"

response = requests.get(url)
print(f"Status: {response.status_code}")

if response.status_code != 200 or "Invalid Key" in response.text:
    print("Census API error. Check your API key.")
    print(response.text[:300])
    exit()

data = response.json()
headers = data[0]
rows = data[1:]

df_census = pd.DataFrame(rows, columns=headers)
df_census.rename(columns={"zip code tabulation area": "zip"}, inplace=True)
df_census = df_census[df_census["zip"].isin(socal_zips)].copy()

for col in VARIABLES:
    df_census[col] = pd.to_numeric(df_census[col], errors="coerce")

# Replace Census null code with NaN
df_census.replace(-666666666, pd.NA, inplace=True)

census_result = pd.DataFrame()
census_result["zip"] = df_census["zip"].values
census_result["total_population"] = df_census["B01003_001E"].values
census_result["youth_pop_5_17"] = (
        df_census["B01001_004E"].values + df_census["B01001_005E"].values +
        df_census["B01001_006E"].values + df_census["B01001_028E"].values +
        df_census["B01001_029E"].values + df_census["B01001_030E"].values
)
census_result["median_household_income"] = df_census["B19013_001E"].values
census_result["total_households"] = df_census["B11005_001E"].values
census_result["households_with_children"] = df_census["B11005_002E"].values
census_result["pct_households_with_children"] = (
        census_result["households_with_children"] /
        census_result["total_households"] * 100
).round(2)

census_result = census_result.sort_values("zip").reset_index(drop=True)
census_result.to_csv("socal_census_data.csv", index=False)

print(f"Zips matched: {len(census_result)} out of {len(socal_zips)}")
missing = set(socal_zips) - set(census_result["zip"].values)
print(f"Zips with no data: {len(missing)} (PO boxes / non-residential)")
print("Saved: socal_census_data.csv")

#distances and time driving calculations
print()
print("=" * 60)
print("STEP 5: Calculating distances and drive times from zip centroids")
print("=" * 60)
print(f"Corona: {CORONA_ADDRESS}")
print(f"Tustin: {TUSTIN_ADDRESS}")
print(f"Laguna Niguel: {LAGUNA_ADDRESS}")
print()

ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_zcta_national.zip"


zcta_df = pd.read_csv(
    ZCTA_URL,
    compression="zip",
    sep="\t",
    dtype={"GEOID": str},
)
zcta_df.columns = zcta_df.columns.str.strip()
zcta_df = zcta_df[["GEOID", "INTPTLAT", "INTPTLONG"]]
zcta_df.columns = ["zip", "lat", "lon"]
zcta_df["zip"] = zcta_df["zip"].str.strip()
# Filter to SoCal zips
zcta_socal = zcta_df[zcta_df["zip"].isin(socal_zips)].copy().reset_index(drop=True)
print(f"Zip centroids found: {len(zcta_socal)} out of {len(socal_zips)}")

def calc_distance(row, facility_coords):
    try:
        return round(geodesic((row["lat"], row["lon"]), facility_coords).miles, 2)
    except:
        return None

zcta_socal["dist_to_corona_mi"] = zcta_socal.apply(calc_distance, axis=1, facility_coords=CORONA_COORDS)
zcta_socal["dist_to_tustin_mi"] = zcta_socal.apply(calc_distance, axis=1, facility_coords=TUSTIN_COORDS)
zcta_socal["dist_to_laguna_mi"] = zcta_socal.apply(calc_distance, axis=1, facility_coords=LAGUNA_COORDS)

# Drive times via OSRM
# OSRM uses real road network data from OpenStreetMap
# Returns drive time in seconds which we convert to minutes
print("Calculating drive times via OSRM (this may take a few minutes)")

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"
def get_drive_time(origin_lat, origin_lon, dest_lat, dest_lon):
    try:
        url = f"{OSRM_BASE}/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok":
                duration_seconds = data["routes"][0]["duration"]
                return round(duration_seconds / 60, 1)
        return None
    except:
        return None

corona_times = []
tustin_times = []
laguna_times = []

total = len(zcta_socal)
for i, row in zcta_socal.iterrows():
    lat, lon = row["lat"], row["lon"]
    corona_times.append(get_drive_time(lat, lon, CORONA_COORDS[0], CORONA_COORDS[1]))
    tustin_times.append(get_drive_time(lat, lon, TUSTIN_COORDS[0], TUSTIN_COORDS[1]))
    laguna_times.append(get_drive_time(lat, lon, LAGUNA_COORDS[0], LAGUNA_COORDS[1]))
    time.sleep(0.2)

    if (i + 1) % 50 == 0:
        print(f"Progress: {i + 1}/{total} zip codes processed")

zcta_socal["drive_time_to_corona_min"] = corona_times
zcta_socal["drive_time_to_tustin_min"] = tustin_times
zcta_socal["drive_time_to_laguna_min"] = laguna_times

zcta_socal = zcta_socal.sort_values("zip").reset_index(drop=True)
zcta_socal.to_csv("socal_zip_distances.csv", index=False)

print("Saved: socal_zip_distances.csv")
print()
print("Sample output:")
print(zcta_socal[[
    "zip",
    "dist_to_corona_mi", "dist_to_tustin_mi", "dist_to_laguna_mi",
    "drive_time_to_corona_min", "drive_time_to_tustin_min", "drive_time_to_laguna_min"
]].head(10).to_string(index=False))


print()
print("=" * 60)
print("ALL STEPS COMPLETE")
print("Output files:")
print("  - corona_all_visits_cleaned.csv")
print("  - tustin_all_visits_cleaned.csv")
print("  - socal_census_data.csv")
print("  - socal_zip_distances.csv")
print("=" * 60)