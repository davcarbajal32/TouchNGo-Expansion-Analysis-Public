import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

output_dir = "goal1_outputs/goal1_outputs"
os.makedirs(output_dir, exist_ok=True)

d2 = pd.read_csv("dataset2_final.csv", low_memory=False)
d3 = pd.read_csv("dataset3_final.csv", low_memory=False)
d3['date'] = pd.to_datetime(d3['date'])

colors = {
    'Corona': '#FF6F00',
    'Tustin': '#1565C0',
    'bar_main': '#1976D2',
    'bar_secondary': '#42A5F5',
    'accent': '#FF6F00',
    'bg': '#F8FAFC',
    'grid': '#E3EAF2'
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': colors['bg'],
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': colors['grid'],
    'grid.linewidth': 0.8,
    'axes.axisbelow': True
})

label_map = {
    'First Touch & Control - Beginners - Youngers': 'First Touch (Beg/Young)',
    'Speed of Play - Advanced - Olders': 'Speed of Play (Adv/Old)',
    'Defense - Youngers': 'Defense (Young)',
    'Ball Mastery - Beginners - Youngers': 'Ball Mastery (Beg/Young)',
    'Shooting & Finishing - Beginners - Youngers': 'Shooting (Beg/Young)',
    'Shooting & Finishing - Intermediate - Youngers': 'Shooting (Int/Young)',
    'Juniors': 'Juniors',
    'Passing & Movement - Intermediate - Youngers': 'Passing (Int/Young)',
    'Private Sessions': 'Private Sessions',
    'Speed of Play - Advanced - Youngers': 'Speed of Play (Adv/Young)',
    'Camp': 'Camp',
    'Defense - Olders': 'Defense (Old)',
    'Ball Mastery - Intermediate - Youngers': 'Ball Mastery (Int/Young)',
    'First Touch & Control - Beginners - Olders': 'First Touch (Beg/Old)',
    'Shooting & Finishing - Intermediate / Advanced - Olders': 'Shooting (Int-Adv/Old)',
    'Strength & Conditioning': 'Strength & Conditioning',
    'Shooting & Finishing - Beginners - Olders': 'Shooting (Beg/Old)',
    'Passing & Movement - Intermediate - Olders': 'Passing (Int/Old)',
}
d3['session_short'] = d3['session_category'].map(label_map).fillna(d3['session_category'])

#visit frequency distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Customer Visit Frequency Distribution', fontsize=16, fontweight='bold')

bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 750]
labels_freq = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '101-200', '200+']

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = d2[d2['facility'] == facility]['total_visits'].dropna()
    counts, _ = np.histogram(sub, bins=bins)
    bars = ax.bar(range(len(labels_freq)), counts,
                  color=[colors['accent'] if c == counts.max() else colors['bar_main'] for c in counts],
                  edgecolor='white', linewidth=0.5, width=0.7)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels_freq)))
    ax.set_xticklabels(labels_freq, fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Total Visits', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/13_visit_frequency_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 13_visit_frequency_distribution.png")

#customer lifetime distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Customer Lifetime — Days Active (First to Last Visit)', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = d2[d2['facility'] == facility]['days_active'].dropna()
    sub = sub[sub > 0]
    ax.hist(sub, bins=30, color=colors['bar_main'], edgecolor='white', linewidth=0.5)
    ax.axvline(sub.median(), color=colors['accent'], linewidth=2,
               linestyle='--', label=f'Median: {sub.median():.0f} days')
    ax.axvline(sub.mean(), color='green', linewidth=2,
               linestyle='--', label=f'Mean: {sub.mean():.0f} days')
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Active', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/14_customer_lifetime.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 14_customer_lifetime.png")

#cohort retention
d3_tustin = d3[d3['facility'] == 'Tustin'].copy()
first_visit = d3_tustin.groupby('client_id')['date'].min().reset_index()
first_visit['join_year'] = first_visit['date'].dt.year
first_visit = first_visit[first_visit['join_year'].between(2019, 2024)]

last_visit = d3_tustin.groupby('client_id')['date'].max().reset_index()
last_visit.columns = ['client_id', 'last_date']
cohort = first_visit.merge(last_visit, on='client_id')

cutoff = pd.Timestamp('2025-12-31')
check_years = [2020, 2021, 2022, 2023, 2024, 2025]
join_years = sorted(first_visit['join_year'].unique())

retention_data = []
for jy in join_years:
    cohort_clients = cohort[cohort['join_year'] == jy]
    n_total = len(cohort_clients)
    row = {'join_year': jy, 'total': n_total}
    for cy in check_years:
        if cy > jy:
            check_date = pd.Timestamp(f'{cy}-12-31')
            if check_date <= cutoff:
                active = cohort_clients[cohort_clients['last_date'] >= pd.Timestamp(f'{cy}-01-01')]
                row[str(cy)] = len(active) / n_total * 100 if n_total > 0 else 0
    retention_data.append(row)

ret_df = pd.DataFrame(retention_data).set_index('join_year')

fig, ax = plt.subplots(figsize=(14, 7))
year_colors_cohort = {
    2019: '#90CAF9', 2020: '#26C6DA', 2021: '#66BB6A',
    2022: '#FFA726', 2023: '#EF5350', 2024: '#AB47BC'
}

for jy in join_years:
    row = ret_df.loc[jy]
    x_vals = []
    y_vals = []
    for cy in check_years:
        if str(cy) in row and not pd.isna(row[str(cy)]):
            x_vals.append(cy)
            y_vals.append(row[str(cy)])
    if x_vals:
        ax.plot(x_vals, y_vals, marker='o', linewidth=2.5, markersize=6,
                label=f'Joined {jy} (n={int(row["total"])})',
                color=year_colors_cohort.get(jy, '#888'))

ax.set_title('Tustin — Cohort Retention by Join Year', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('% Still Active', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xticks(check_years)

plt.tight_layout()
plt.savefig(f"{output_dir}/15_cohort_retention.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 15_cohort_retention.png")

#first session type vs avg total visits
first_sess = d3.sort_values('date').groupby(['client_id','facility']).first().reset_index()[['client_id','facility','session_category']]
first_sess['session_short'] = first_sess['session_category'].map(label_map).fillna(first_sess['session_category'])
merged = first_sess.merge(d2[['client_id','facility','total_visits']], on=['client_id','facility'])

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('First Session Type vs Average Total Visits', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = merged[merged['facility'] == facility].groupby('session_short')['total_visits'].mean().sort_values(ascending=True)
    colors_list = [colors['accent'] if v == sub.max() else colors['bar_main'] for v in sub]
    bars = ax.barh(sub.index, sub.values, color=colors_list, edgecolor='white', linewidth=0.5, height=0.7)
    for bar, val in zip(bars, sub.values):
        ax.text(bar.get_width() + sub.max()*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', ha='left', fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Average Total Visits', fontsize=11)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, sub.max() * 1.15)

plt.tight_layout()
plt.savefig(f"{output_dir}/16_first_session_vs_visits.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 16_first_session_vs_visits.png")

#session popularity over time
d3['age_group'] = d3['session_category'].apply(
    lambda x: 'Youngers' if 'Younger' in str(x) else ('Olders' if 'Older' in str(x) else 'Other'))

tustin_time = d3[(d3['facility']=='Tustin') & (d3['year']>=2019)]
age_time = tustin_time.groupby(['year','age_group']).size().reset_index(name='visits')
age_pivot = age_time.pivot(index='year', columns='age_group', values='visits').fillna(0)

fig, ax = plt.subplots(figsize=(14, 6))
age_palette = {'Youngers': colors['Tustin'], 'Olders': colors['accent'], 'Other': '#78909C'}

for col in age_pivot.columns:
    ax.plot(age_pivot.index, age_pivot[col], marker='o', linewidth=2.5,
            label=col, color=age_palette.get(col, '#888'), markersize=6)

ax.set_title('Tustin — Youngers vs Olders vs Other Visits Over Time', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Visits', fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_xticks(age_pivot.index)

plt.tight_layout()
plt.savefig(f"{output_dir}/17_session_mix_over_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 17_session_mix_over_time.png")

#return rate by session type
ret_sess = d3.groupby(['facility','session_short'])['returned_within_30_days'].mean().reset_index()
ret_sess['return_pct'] = ret_sess['returned_within_30_days'] * 100

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('30-Day Return Rate by Session Type', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = ret_sess[ret_sess['facility']==facility].sort_values('return_pct', ascending=True)
    colors_list = [colors['accent'] if v == sub['return_pct'].max() else colors['bar_main'] for v in sub['return_pct']]
    bars = ax.barh(sub['session_short'], sub['return_pct'], color=colors_list,
                   edgecolor='white', linewidth=0.5, height=0.7)
    for bar, val in zip(bars, sub['return_pct']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('30-Day Return Rate (%)', fontsize=11)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, 110)

plt.tight_layout()
plt.savefig(f"{output_dir}/18_return_rate_by_session.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 18_return_rate_by_session.png")

#return rate by coach
ret_coach = d3.groupby(['facility','coach'])['returned_within_30_days'].agg(['mean','count']).reset_index()
ret_coach.columns = ['facility','coach','return_rate','visit_count']
ret_coach['return_pct'] = ret_coach['return_rate'] * 100
ret_coach = ret_coach[ret_coach['visit_count'] >= 50]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('30-Day Return Rate by Coach (min 50 visits)', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = ret_coach[ret_coach['facility']==facility].sort_values('return_pct', ascending=True)
    colors_list = [colors['accent'] if v == sub['return_pct'].max() else colors['bar_main'] for v in sub['return_pct']]
    bars = ax.barh(sub['coach'], sub['return_pct'], color=colors_list,
                   edgecolor='white', linewidth=0.5, height=0.7)
    for bar, val, n in zip(bars, sub['return_pct'], sub['visit_count']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}% (n={n:,})', va='center', ha='left', fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('30-Day Return Rate (%)', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim(0, 115)

plt.tight_layout()
plt.savefig(f"{output_dir}/19_return_rate_by_coach.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 19_return_rate_by_coach.png")

#drive time distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Drive Time to Attended Facility — Customer Distribution', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = d2[d2['facility']==facility]['drive_time_to_attended_facility'].dropna()
    ax.hist(sub, bins=30, color=colors['bar_main'], edgecolor='white', linewidth=0.5)
    ax.axvline(sub.median(), color=colors['accent'], linewidth=2,
               linestyle='--', label=f'Median: {sub.median():.1f} min')
    ax.axvline(sub.mean(), color='green', linewidth=2,
               linestyle='--', label=f'Mean: {sub.mean():.1f} min')
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Drive Time (minutes)', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/20_drive_time_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 20_drive_time_distribution.png")

#drive time vs total visits scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Drive Time vs Total Visits Per Customer', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = d2[d2['facility']==facility][['drive_time_to_attended_facility','total_visits']].dropna()
    sub = sub[sub['total_visits'] <= 200]
    ax.scatter(sub['drive_time_to_attended_facility'], sub['total_visits'],
               alpha=0.3, s=15, color=colors['bar_main'], edgecolor='none')
    z = np.polyfit(sub['drive_time_to_attended_facility'], sub['total_visits'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sub['drive_time_to_attended_facility'].min(),
                         sub['drive_time_to_attended_facility'].max(), 100)
    ax.plot(x_line, p(x_line), color=colors['accent'], linewidth=2.5, label='Trend line')
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Drive Time to Facility (minutes)', fontsize=11)
    ax.set_ylabel('Total Visits', fontsize=11)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/21_drive_time_vs_visits.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 21_drive_time_vs_visits.png")

#day of week x hour heatmap
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
hour_label = {1:'1PM',2:'2PM',3:'3PM',4:'4PM',5:'5PM',6:'6PM',7:'7PM',
              8:'8PM',9:'9AM',10:'10AM',11:'11AM',12:'12PM'}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Visit Heatmap — Day of Week vs Hour of Day', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = d3[d3['facility']==facility].copy()
    sub['hour_label'] = sub['hour'].map(hour_label).fillna(sub['hour'].astype(str))
    pivot = sub.groupby(['day_of_week','hour_label']).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order, fontsize=10)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Visits')

plt.tight_layout()
plt.savefig(f"{output_dir}/22_day_hour_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 22_day_hour_heatmap.png")

#month joined vs churn rate
d3_first = d3.sort_values('date').groupby(['client_id','facility'])['month'].first().reset_index()
d3_first.columns = ['client_id','facility','join_month']
merged_churn = d3_first.merge(d2[['client_id','facility','is_churned']], on=['client_id','facility'])

churn_by_month = merged_churn.groupby(['facility','join_month'])['is_churned'].mean().reset_index()
churn_by_month['churn_pct'] = churn_by_month['is_churned'] * 100
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
churn_by_month['month_name'] = churn_by_month['join_month'].map(month_names)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Churn Rate by Month of First Enrollment', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = churn_by_month[churn_by_month['facility']==facility].sort_values('join_month')
    bars = ax.bar(sub['month_name'], sub['churn_pct'],
                  color=[colors['accent'] if v == sub['churn_pct'].max() else colors['bar_main']
                         for v in sub['churn_pct']],
                  edgecolor='white', linewidth=0.5, width=0.7)
    for bar, val in zip(bars, sub['churn_pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Month of First Visit', fontsize=11)
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_ylim(0, sub['churn_pct'].max() * 1.15)

plt.tight_layout()
plt.savefig(f"{output_dir}/23_churn_by_enrollment_month.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 23_churn_by_enrollment_month.png")

print(f"\nAll additional charts saved to {output_dir}/")