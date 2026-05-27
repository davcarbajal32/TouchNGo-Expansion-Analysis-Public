# Model D - Goal 3
# Huff spatial interaction model
# Estimates choice probability per ZIP: P(Corona) vs P(Tustin) vs P(Laguna)
# Used to estimate cannibalization from existing locations to Laguna
#
# NOT a learned model - this is a formula-based spatial choice model
# Probability of choosing location i from ZIP j:
#   P_ij = (A_i / D_ij^beta) / sum_k(A_k / D_kj^beta)
# where A_i = attractiveness of location i, D_ij = drive time from ZIP j to i

import pandas as pd
import numpy as np
import os

# load data
print("=" * 60)
print("MODEL D - Huff Spatial Choice Model")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
d1 = pd.read_csv(os.path.join(script_dir, "dataset1_final.csv"))
d2 = pd.read_csv(os.path.join(script_dir, "dataset2_final.csv"))
d3 = pd.read_csv(os.path.join(script_dir, "dataset3_final.csv"))
d3['date'] = pd.to_datetime(d3['date'])

print("DS1 rows:", len(d1))


# build ZIP panel: 2022+ visit/customer counts per facility per ZIP
zip_lookup = d2[['client_id', 'facility', 'zip']].drop_duplicates(subset=['client_id', 'facility'])
d3_z = d3.merge(zip_lookup, on=['client_id', 'facility'], how='left')
d3_z['zip_int'] = pd.to_numeric(d3_z['zip'], errors='coerce')
d3_filt = d3_z[(d3_z['status'] == 'Signed in') &
               (d3_z['date'] >= '2022-01-01')].dropna(subset=['zip_int'])
d3_filt['zip_int'] = d3_filt['zip_int'].astype(int)

# visits per ZIP per facility
zf = d3_filt.groupby(['zip_int', 'facility']).size().reset_index(name='visits')
zf_wide = zf.pivot(index='zip_int', columns='facility', values='visits').fillna(0).reset_index()
zf_wide.columns.name = None
zf_wide = zf_wide.rename(columns={'Corona': 'corona_visits_2022', 'Tustin': 'tustin_visits_2022'})

d1_features = d1[['zip', 'drive_time_to_corona_min', 'drive_time_to_tustin_min',
                  'drive_time_to_laguna_min']].rename(columns={'zip': 'zip_int'})
zp = d1_features.merge(zf_wide, on='zip_int', how='left').fillna(0)


# attractiveness proxy: total 2022+ visit volume at each existing location
# this approximates location "size" or "draw power"
A_corona = zp['corona_visits_2022'].sum()
A_tustin = zp['tustin_visits_2022'].sum()

print("Corona attractiveness (2022+ visits):", int(A_corona))
print("Tustin attractiveness (2022+ visits):", int(A_tustin))


# beta parameter controls distance decay
# beta = 1.5 to 2.0 is typical for retail/service
# higher beta means people are less willing to travel far
# we will try a few values for sensitivity analysis
BETA = 2.0

# for Laguna we test 3 scenarios since the location does not exist yet
# Scenario 1: Laguna is same size as Corona
# Scenario 2: Laguna is between Corona and Tustin (geometric mean)
# Scenario 3: Laguna is same size as Tustin
A_laguna_scenarios = {
    'Laguna_as_Corona': A_corona,
    'Laguna_midpoint': np.sqrt(A_corona * A_tustin),
    'Laguna_as_Tustin': A_tustin
}


# compute Huff probabilities per ZIP for each scenario
results = []

for scenario_name, A_laguna in A_laguna_scenarios.items():
    df = zp.copy()
    df['A_corona_over_D'] = A_corona / (df['drive_time_to_corona_min'] ** BETA)
    df['A_tustin_over_D'] = A_tustin / (df['drive_time_to_tustin_min'] ** BETA)
    df['A_laguna_over_D'] = A_laguna / (df['drive_time_to_laguna_min'] ** BETA)
    df['denom'] = df['A_corona_over_D'] + df['A_tustin_over_D'] + df['A_laguna_over_D']

    df['p_corona'] = df['A_corona_over_D'] / df['denom']
    df['p_tustin'] = df['A_tustin_over_D'] / df['denom']
    df['p_laguna'] = df['A_laguna_over_D'] / df['denom']

    # cannibalization: existing visits expected to shift to Laguna
    # this is the proportion of each ZIP's existing visits that the Huff model
    # says would now go to Laguna instead
    #
    # IMPORTANT - we apply two filters to make this defensible:
    #   1. P(Laguna) must be >= 30% (ignores low-probability noise)
    #   2. Laguna must be at most 10 min farther than the existing facility
    #      (a customer driving to Corona will not switch to a much-farther Laguna
    #       even if the Huff formula assigns it some probability)
    # the 10-min tolerance allows for real-world reasons people drive a bit
    # farther: traffic, coach loyalty, schedule, sibling overlap, etc.
    THRESHOLD = 0.30
    TOLERANCE = 10

    df['corona_allowed'] = ((df['p_laguna'] >= THRESHOLD) &
                            (df['drive_time_to_laguna_min'] <= df['drive_time_to_corona_min'] + TOLERANCE))
    df['tustin_allowed'] = ((df['p_laguna'] >= THRESHOLD) &
                            (df['drive_time_to_laguna_min'] <= df['drive_time_to_tustin_min'] + TOLERANCE))

    df['corona_visits_lost_to_laguna'] = np.where(
        df['corona_allowed'], df['corona_visits_2022'] * df['p_laguna'], 0
    )
    df['tustin_visits_lost_to_laguna'] = np.where(
        df['tustin_allowed'], df['tustin_visits_2022'] * df['p_laguna'], 0
    )
    df['total_cannibalized_visits'] = df['corona_visits_lost_to_laguna'] + df['tustin_visits_lost_to_laguna']

    results.append({
        'scenario': scenario_name,
        'A_laguna': A_laguna,
        'mean_p_laguna_near': df.loc[df['drive_time_to_laguna_min'] <= 30, 'p_laguna'].mean(),
        'mean_p_laguna_all': df['p_laguna'].mean(),
        'total_cannibalized_visits': df['total_cannibalized_visits'].sum(),
        'corona_visits_lost': df['corona_visits_lost_to_laguna'].sum(),
        'tustin_visits_lost': df['tustin_visits_lost_to_laguna'].sum(),
        'df': df
    })


# print summary across scenarios
print("=" * 60)
print("HUFF MODEL - CANNIBALIZATION SCENARIOS")
print("=" * 60)
print("Beta (distance decay exponent):", BETA)

summary = pd.DataFrame([{
    'Scenario': r['scenario'],
    'A_laguna': int(r['A_laguna']),
    'Mean P(Laguna) near (<30 min)': round(r['mean_p_laguna_near'], 3),
    'Mean P(Laguna) all ZIPs': round(r['mean_p_laguna_all'], 3),
    'Corona visits lost': round(r['corona_visits_lost'], 0),
    'Tustin visits lost': round(r['tustin_visits_lost'], 0),
    'Total cannibalized': round(r['total_cannibalized_visits'], 0)
} for r in results])
print(summary.to_string(index=False))


# show the midpoint scenario in detail (most realistic)
print("=" * 60)
print("DETAIL: MIDPOINT SCENARIO (A_laguna = geometric mean)")
print("=" * 60)
mid_df = results[1]['df'].copy()

# top 15 ZIPs by Laguna probability (where most demand will come from)
top_laguna = mid_df.sort_values('p_laguna', ascending=False).head(15)
print("Top 15 ZIPs by predicted P(Laguna):")
print(top_laguna[['zip_int', 'drive_time_to_corona_min', 'drive_time_to_tustin_min',
                  'drive_time_to_laguna_min', 'p_corona', 'p_tustin', 'p_laguna',
                  'corona_visits_2022', 'tustin_visits_2022',
                  'total_cannibalized_visits']].round(3).to_string(index=False))


# sensitivity analysis: different beta values
print("=" * 60)
print("SENSITIVITY: BETA VALUES (midpoint scenario)")
print("=" * 60)
A_laguna_mid = A_laguna_scenarios['Laguna_midpoint']

for beta_test in [1.0, 1.5, 2.0, 2.5, 3.0]:
    df = zp.copy()
    df['A_corona_over_D'] = A_corona / (df['drive_time_to_corona_min'] ** beta_test)
    df['A_tustin_over_D'] = A_tustin / (df['drive_time_to_tustin_min'] ** beta_test)
    df['A_laguna_over_D'] = A_laguna_mid / (df['drive_time_to_laguna_min'] ** beta_test)
    df['denom'] = df['A_corona_over_D'] + df['A_tustin_over_D'] + df['A_laguna_over_D']
    df['p_laguna'] = df['A_laguna_over_D'] / df['denom']

    # apply same threshold + tolerance rule
    cor_ok = (df['p_laguna'] >= 0.30) & (df['drive_time_to_laguna_min'] <= df['drive_time_to_corona_min'] + 10)
    tus_ok = (df['p_laguna'] >= 0.30) & (df['drive_time_to_laguna_min'] <= df['drive_time_to_tustin_min'] + 10)
    cor_lost = np.where(cor_ok, df['corona_visits_2022'] * df['p_laguna'], 0).sum()
    tus_lost = np.where(tus_ok, df['tustin_visits_2022'] * df['p_laguna'], 0).sum()
    near_p = df.loc[df['drive_time_to_laguna_min'] <= 30, 'p_laguna'].mean()
    print(f"beta = {beta_test}: mean P(Laguna) near = {round(near_p, 3)}, total cannibalized = {round(cor_lost + tus_lost, 0)}")


# save output for Model E to use
mid_df.to_csv(os.path.join(script_dir, 'huff_probabilities.csv'), index=False)
print("=" * 60)
print("Saved huff_probabilities.csv for Model E to consume")
print("=" * 60)
print("DONE")
print("=" * 60)