import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import os
import json
import urllib.request

output_dir = "goal1_outputs"
os.makedirs(output_dir, exist_ok=True)

d1 = pd.read_csv("dataset1_final.csv")
d2 = pd.read_csv("dataset2_final.csv", low_memory=False)
d3 = pd.read_csv("dataset3_final.csv", low_memory=False)

d1['zip'] = d1['zip'].astype(str).str.zfill(5)

#heatmap settings
HEAT_RADIUS = 40
HEAT_BLUR = 35
HEAT_MIN_OPACITY = 0.4

#load zip centroids from socal_zip_distances if it exists
def load_centroids():
    if os.path.exists("socal_zip_distances.csv"):
        df = pd.read_csv("socal_zip_distances.csv")
        if 'lat' in df.columns and 'lon' in df.columns:
            df['zip'] = df['zip'].astype(str).str.zfill(5)
            return dict(zip(df['zip'], zip(df['lat'], df['lon'])))

    print("socal_zip_distances.csv not found, downloading centroids...")
    try:
        url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ca_california_zip_codes_geo.min.json"
        req = urllib.request.Request(url, headers={'User-Agent': 'TNG/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            geo = json.loads(resp.read())
        centroids = {}
        for feat in geo['features']:
            z = str(feat['properties'].get('ZCTA5CE10', feat['properties'].get('ZCTA5', ''))).zfill(5)
            coords = feat['geometry']['coordinates']
            if feat['geometry']['type'] == 'Polygon':
                pts = np.array(coords[0])
            else:
                pts = np.array(coords[0][0])
            centroids[z] = (float(pts[:,1].mean()), float(pts[:,0].mean()))
        return centroids
    except Exception as e:
        print(f"Could not load centroids: {e}")
        return {}

print("Loading ZIP code centroids...")
centroids = load_centroids()
print(f"Loaded {len(centroids)} centroids")

d1['lat'] = d1['zip'].map(lambda z: centroids.get(z, (None, None))[0])
d1['lon'] = d1['zip'].map(lambda z: centroids.get(z, (None, None))[1])
d1_geo = d1.dropna(subset=['lat', 'lon']).copy()

locations = {
    'Corona': (33.8617, -117.5748),
    'Tustin': (33.7297, -117.7946),
    'Laguna Niguel': (33.5225, -117.7075)
}

def base_map(zoom=10):
    m = folium.Map(location=(33.75, -117.75), zoom_start=zoom, tiles='CartoDB positron')
    for name, coords in locations.items():
        icon_color = 'red' if name == 'Corona' else ('blue' if name == 'Tustin' else 'green')
        folium.Marker(coords, popup=f"Touch N Go - {name}", tooltip=name,
                      icon=folium.Icon(color=icon_color, icon='star', prefix='fa')).add_to(m)
    return m

def make_heatmap(data, value_col, title, filename):
    m = base_map()

    #filter to valid positive values
    valid = data[(data[value_col].notna()) & (data[value_col] > 0)].copy()

    if len(valid) > 0:
        #normalize values to 0-1 scale so heat gradient applies evenly
        v_min = valid[value_col].min()
        v_max = valid[value_col].max()
        if v_max > v_min:
            valid['normalized'] = (valid[value_col] - v_min) / (v_max - v_min)
        else:
            valid['normalized'] = 1.0

        #apply slight exponential boost so mid-range values show up more clearly
        valid['normalized'] = np.power(valid['normalized'], 0.6)

        heat_data = [[r['lat'], r['lon'], float(r['normalized'])]
                     for _, r in valid.iterrows()]

        HeatMap(heat_data,
                radius=HEAT_RADIUS,
                blur=HEAT_BLUR,
                min_opacity=HEAT_MIN_OPACITY,
                max_zoom=13,
                gradient={
                    '0.0': '#0000FF',
                    '0.25': '#00FFFF',
                    '0.5': '#00FF00',
                    '0.75': '#FFFF00',
                    '1.0': '#FF0000'
                }).add_to(m)

    title_html = f'<h3 style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1000;background:white;padding:8px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:sans-serif;">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(f"{output_dir}/{filename}")
    print(f"Saved {filename}")

#total customers
make_heatmap(d1_geo, 'total_customers', 'Total Customers by ZIP Code', 'map_01_total_customers.html')

#total visits
make_heatmap(d1_geo, 'total_visits', 'Total Visits by ZIP Code', 'map_02_total_visits.html')

#customers per 1000 youth
make_heatmap(d1_geo, 'customers_per_1000_youth', 'Market Penetration - Customers per 1,000 Youth', 'map_03_market_penetration.html')

#corona customers only
make_heatmap(d1_geo, 'corona_total_customers', 'Corona - Customers by ZIP Code', 'map_04_corona_customers.html')

#tustin customers only
make_heatmap(d1_geo, 'tustin_total_customers', 'Tustin - Customers by ZIP Code', 'map_05_tustin_customers.html')

#active members
d1_geo['total_active_members'] = d1_geo['corona_active_members'].fillna(0) + d1_geo['tustin_active_members'].fillna(0)
make_heatmap(d1_geo, 'total_active_members', 'Active Members by ZIP Code', 'map_06_active_members.html')

#churned customers
d1_geo['total_churned'] = d1_geo['corona_churned_customers'].fillna(0) + d1_geo['tustin_churned_customers'].fillna(0)
make_heatmap(d1_geo, 'total_churned', 'Churned Customers by ZIP Code', 'map_07_churned_customers.html')

#corona retention rate
corona_ret = d1_geo[d1_geo['corona_total_customers'] >= 3].copy()
make_heatmap(corona_ret, 'corona_retention_rate', 'Corona - Retention Rate by ZIP Code', 'map_08_corona_retention.html')

#tustin retention rate
tustin_ret = d1_geo[d1_geo['tustin_total_customers'] >= 5].copy()
make_heatmap(tustin_ret, 'tustin_retention_rate', 'Tustin - Retention Rate by ZIP Code', 'map_09_tustin_retention.html')

#youth population
make_heatmap(d1_geo, 'youth_pop_5_17', 'Youth Population Ages 5 to 17 by ZIP Code', 'map_10_youth_population.html')

#median household income
make_heatmap(d1_geo[d1_geo['median_household_income'] > 0], 'median_household_income', 'Median Household Income by ZIP Code', 'map_11_median_income.html')

#youngers vs olders visits by zip
d2['client_id'] = d2['client_id'].astype(str)
d3['client_id'] = d3['client_id'].astype(str)
d3_zip = d3.merge(d2[['client_id', 'facility', 'zip']], on=['client_id', 'facility'], how='left')
d3_zip['zip'] = d3_zip['zip'].astype(str).str.zfill(5)
d3_zip['lat'] = d3_zip['zip'].map(lambda z: centroids.get(z, (None, None))[0])
d3_zip['lon'] = d3_zip['zip'].map(lambda z: centroids.get(z, (None, None))[1])
d3_zip = d3_zip.dropna(subset=['lat', 'lon'])

d3_zip['is_youngers'] = d3_zip['session_category'].str.contains('Younger', na=False).astype(int)
d3_zip['is_olders'] = d3_zip['session_category'].str.contains('Older', na=False).astype(int)

zip_age = d3_zip.groupby('zip').agg(
    lat=('lat', 'first'),
    lon=('lon', 'first'),
    youngers_visits=('is_youngers', 'sum'),
    olders_visits=('is_olders', 'sum')
).reset_index()

#youngers visits
make_heatmap(zip_age, 'youngers_visits', 'Youngers Training Sessions - Visit Density by ZIP', 'map_12_youngers_visits.html')

#olders visits
make_heatmap(zip_age, 'olders_visits', 'Olders Training Sessions - Visit Density by ZIP', 'map_13_olders_visits.html')

#membership conversion rate
d1_geo['avg_conversion'] = (
    (d1_geo['corona_membership_conversion_rate'].fillna(0) * d1_geo['corona_total_customers'].fillna(0) +
     d1_geo['tustin_membership_conversion_rate'].fillna(0) * d1_geo['tustin_total_customers'].fillna(0)) /
    (d1_geo['total_customers'].replace(0, np.nan))
)
make_heatmap(d1_geo[d1_geo['total_customers'] >= 3], 'avg_conversion', 'Membership Conversion Rate by ZIP Code', 'map_14_membership_conversion.html')

#average visits per customer
d1_geo['avg_visits_overall'] = (
    (d1_geo['corona_avg_visits_per_customer'].fillna(0) * d1_geo['corona_total_customers'].fillna(0) +
     d1_geo['tustin_avg_visits_per_customer'].fillna(0) * d1_geo['tustin_total_customers'].fillna(0)) /
    (d1_geo['total_customers'].replace(0, np.nan))
)
make_heatmap(d1_geo[d1_geo['total_customers'] >= 3], 'avg_visits_overall', 'Average Visits Per Customer by ZIP Code', 'map_15_avg_visits_per_customer.html')

print(f"\nAll 15 heatmaps saved to {output_dir}/")
print("Open each HTML file in a browser to view.")