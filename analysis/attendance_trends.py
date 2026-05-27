import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

output_dir = "goal1_outputs"
os.makedirs(output_dir, exist_ok=True)

d3 = pd.read_csv("dataset3_final.csv", low_memory=False)
d3['date'] = pd.to_datetime(d3['date'])

colors = {
    'Corona': '#FF6F00',
    'Tustin': '#1565C0',
    'grid': '#E3EAF2',
    'bg': '#F8FAFC'
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

#monthly attendance for both locations
monthly = d3.groupby(['facility', 'year', 'month']).size().reset_index(name='visits')
monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

fig, ax = plt.subplots(figsize=(18, 7))

for facility, color in [('Tustin', colors['Tustin']), ('Corona', colors['Corona'])]:
    sub = monthly[monthly['facility'] == facility].sort_values('date')
    ax.plot(sub['date'], sub['visits'], color=color, linewidth=2.5,
            label=facility, marker='o', markersize=3)
    ax.fill_between(sub['date'], sub['visits'], alpha=0.08, color=color)

ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'),
           alpha=0.15, color='red', label='COVID-19 Shutdown')
ax.annotate('COVID-19', xy=(pd.Timestamp('2020-03-15'), 0),
            xytext=(pd.Timestamp('2020-06-01'), 400),
            fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

ax.annotate('Corona opens\nOct 2024', xy=(pd.Timestamp('2024-11-01'), 642),
            xytext=(pd.Timestamp('2023-06-01'), 900),
            fontsize=9, color=colors['Corona'],
            arrowprops=dict(arrowstyle='->', color=colors['Corona'], lw=1.2))

ax.set_title('Monthly Visit Attendance - Corona & Tustin (All Time)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Total Visits', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha='right')
ax.legend(fontsize=11, framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/07_monthly_attendance_alltime.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_monthly_attendance_alltime.png")

#year over year monthly visits for tustin
tustin = monthly[monthly['facility'] == 'Tustin'].copy()
tustin_pivot = tustin.pivot_table(index='month', columns='year', values='visits', aggfunc='sum')

fig, ax = plt.subplots(figsize=(14, 7))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
blue_purple_palette = [
    '#BBDEFB', # 2018: Lightest Blue
    '#64B5F6', # 2019: Light Blue
    '#1E88E5', # 2020: Medium Blue
    '#0D47A1', # 2021: Dark Blue
    '#E1BEE7', # 2022: Very Light Purple
    '#BA68C8', # 2023: Light Purple
    '#8E24AA', # 2024: Medium Purple
    '#6A1B9A', # 2025: Dark Purple
    '#4A148C'  # 2026: Deepest Purple
]

years = sorted(tustin_pivot.columns)
for i, (col, color) in enumerate(zip(years, blue_purple_palette)):
    vals = tustin_pivot[col].reindex(range(1,13))
    ax.plot(month_names, vals, marker='o', linewidth=2.5, label=str(int(col)),
            color=color, markersize=6)

ax.set_title('Tustin - Year Over Year Monthly Visits', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Total Visits', fontsize=12)
ax.legend(title='Year', fontsize=10, framealpha=0.9, ncol=2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/08_tustin_yoy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_tustin_yoy.png")

#corona monthly visits since opening
corona = monthly[monthly['facility'] == 'Corona'].sort_values('date')

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(corona['date'], corona['visits'], color=colors['Corona'],
       width=20, edgecolor='white', linewidth=0.5, alpha=0.85)
ax.plot(corona['date'], corona['visits'], color='#BF360C', linewidth=2,
        marker='o', markersize=5)

ax.set_title('Corona - Monthly Attendance Since Opening (Oct 2024)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Total Visits', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

for x, y in zip(corona['date'], corona['visits']):
    ax.text(x, y + 20, f'{y:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/09_corona_monthly.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 09_corona_monthly.png")

#new customers per month at tustin
d3_tustin = d3[d3['facility'] == 'Tustin'].copy()
first_visit = d3_tustin.groupby('client_id')['date'].min().reset_index()
first_visit['year'] = first_visit['date'].dt.year
first_visit['month'] = first_visit['date'].dt.month
first_visit['cohort_date'] = pd.to_datetime(first_visit[['year','month']].assign(day=1))

new_customers = first_visit.groupby('cohort_date').size().reset_index(name='new_customers')
new_customers = new_customers[new_customers['cohort_date'] >= '2019-01-01']

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(new_customers['cohort_date'], new_customers['new_customers'],
       color=colors['Tustin'], width=20, edgecolor='white', linewidth=0.5, alpha=0.85)

ma = new_customers['new_customers'].rolling(3).mean()
ax.plot(new_customers['cohort_date'], ma, color=colors['Corona'],
        linewidth=2.5, label='3-month rolling avg')

ax.set_title('Tustin - New Customers Per Month', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('New Customers', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha='right')
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(f"{output_dir}/10_tustin_new_customers.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 10_tustin_new_customers.png")

#seasonality at tustin
tustin_2022 = d3[(d3['facility']=='Tustin') & (d3['year']>=2022)]
season = tustin_2022.groupby('month_name').size().reset_index(name='visits')
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
season = season.set_index('month_name').reindex(month_order).reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(12), season['visits'],
              color=[colors['Tustin'] if v < season['visits'].max() else colors['Corona']
                     for v in season['visits']],
              edgecolor='white', linewidth=0.5, width=0.7)
ax.set_xticks(range(12))
ax.set_xticklabels([m[:3] for m in month_order], fontsize=11)
ax.set_title('Tustin - Seasonality: Total Monthly Visits (2022 to Present)', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Total Visits', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

for bar, val in zip(bars, season['visits']):
    if not pd.isna(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(val):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/11_seasonality.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 11_seasonality.png")

#30 day return rate over time
d3['year_month'] = d3['date'].dt.to_period('M')
return_rate = (d3[d3['year'] >= 2022]
               .groupby(['facility', 'year_month'])['returned_within_30_days']
               .mean()
               .reset_index())
return_rate['date'] = return_rate['year_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(16, 6))
for facility, color in [('Tustin', colors['Tustin']), ('Corona', colors['Corona'])]:
    sub = return_rate[return_rate['facility'] == facility].sort_values('date')
    ax.plot(sub['date'], sub['returned_within_30_days'] * 100,
            color=color, linewidth=2.5, label=facility, marker='o', markersize=3)

ax.axhline(85, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.set_title('30-Day Return Rate Over Time', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('% of Visits Where Customer Returned Within 30 Days', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f"{output_dir}/12_return_rate_over_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 12_return_rate_over_time.png")

print(f"\nAll attendance trend charts saved to {output_dir}/")