import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "goal1_outputs"
os.makedirs(output_dir, exist_ok=True)

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

#shorten long session labels
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

#total visits by session
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Session Type Popularity - Total Visits', fontsize=18, fontweight='bold', y=1.01)

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    data = (d3[d3['facility'] == facility]
            .groupby('session_short').size()
            .sort_values(ascending=True)
            .reset_index(name='visits'))

    colors_list = [colors['accent'] if v == data['visits'].max() else colors['bar_main']
                   for v in data['visits']]

    bars = ax.barh(data['session_short'], data['visits'], color=colors_list,
                   edgecolor='white', linewidth=0.5, height=0.7)

    for bar, val in zip(bars, data['visits']):
        ax.text(bar.get_width() + data['visits'].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=9, color='#333')

    ax.set_title(facility, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Total Visits', fontsize=11)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, data['visits'].max() * 1.15)

plt.tight_layout()
plt.savefig(f"{output_dir}/01_session_popularity_visits.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_session_popularity_visits.png")

#unique customers per session
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Session Type Popularity - Unique Customers', fontsize=18, fontweight='bold', y=1.01)

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    data = (d3[d3['facility'] == facility]
            .groupby('session_short')['client_id'].nunique()
            .sort_values(ascending=True)
            .reset_index(name='customers'))

    colors_list = [colors['accent'] if v == data['customers'].max() else colors['bar_secondary']
                   for v in data['customers']]

    bars = ax.barh(data['session_short'], data['customers'], color=colors_list,
                   edgecolor='white', linewidth=0.5, height=0.7)

    for bar, val in zip(bars, data['customers']):
        ax.text(bar.get_width() + data['customers'].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=9, color='#333')

    ax.set_title(facility, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Unique Customers', fontsize=11)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, data['customers'].max() * 1.15)

plt.tight_layout()
plt.savefig(f"{output_dir}/02_session_popularity_customers.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_session_popularity_customers.png")

#youngers vs olders breakdown
d3['age_group'] = d3['session_category'].apply(
    lambda x: 'Youngers' if 'Younger' in str(x) else
              ('Olders' if 'Older' in str(x) else 'Other'))

age_data = d3.groupby(['facility', 'age_group']).size().reset_index(name='visits')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Training Volume - Youngers vs Olders vs Other', fontsize=16, fontweight='bold')

age_colors = {'Youngers': '#1976D2', 'Olders': '#FF6F00', 'Other': '#78909C'}

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = age_data[age_data['facility'] == facility]
    total = sub['visits'].sum()
    wedges, texts, autotexts = ax.pie(
        sub['visits'],
        labels=sub['age_group'],
        colors=[age_colors[g] for g in sub['age_group']],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title(f'{facility} ({total:,} total visits)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/03_youngers_vs_olders.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_youngers_vs_olders.png")

#coach popularity
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Coach Popularity - Total Visits Led', fontsize=18, fontweight='bold', y=1.01)

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    data = (d3[d3['facility'] == facility]
            .groupby('coach').size()
            .sort_values(ascending=True)
            .tail(12)
            .reset_index(name='visits'))

    colors_list = [colors['accent'] if v == data['visits'].max() else colors['bar_main']
                   for v in data['visits']]

    bars = ax.barh(data['coach'], data['visits'], color=colors_list,
                   edgecolor='white', linewidth=0.5, height=0.7)

    for bar, val in zip(bars, data['visits']):
        ax.text(bar.get_width() + data['visits'].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=9, color='#333')

    ax.set_title(f'{facility} - Top Coaches', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Total Visits Led', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim(0, data['visits'].max() * 1.15)

plt.tight_layout()
plt.savefig(f"{output_dir}/04_coach_popularity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_coach_popularity.png")

#visits by day of week
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = d3.groupby(['facility','day_of_week']).size().reset_index(name='visits')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Visits by Day of Week', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = dow[dow['facility'] == facility].set_index('day_of_week').reindex(day_order).reset_index()
    bars = ax.bar(sub['day_of_week'], sub['visits'],
                  color=[colors['accent'] if v == sub['visits'].max() else colors['bar_main'] for v in sub['visits']],
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, sub['visits']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sub['visits'].max() * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Visits', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, sub['visits'].max() * 1.12)

plt.tight_layout()
plt.savefig(f"{output_dir}/05_day_of_week.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_day_of_week.png")

#visits by hour of day
hour_label = {1:'1PM',2:'2PM',3:'3PM',4:'4PM',5:'5PM',6:'6PM',7:'7PM',
              8:'8PM',9:'9AM',10:'10AM',11:'11AM',12:'12PM'}

hour_data = d3.groupby(['facility','hour']).size().reset_index(name='visits')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Visits by Hour of Day', fontsize=16, fontweight='bold')

for ax, facility in zip(axes, ['Corona', 'Tustin']):
    sub = hour_data[hour_data['facility'] == facility].copy()
    sub['hour_label'] = sub['hour'].map(hour_label).fillna(sub['hour'].astype(str))
    bars = ax.bar(sub['hour_label'], sub['visits'],
                  color=[colors['accent'] if v == sub['visits'].max() else colors['bar_main'] for v in sub['visits']],
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, sub['visits']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sub['visits'].max() * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    ax.set_title(facility, fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=11)
    ax.set_ylabel('Visits', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{output_dir}/06_hour_of_day.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_hour_of_day.png")

print(f"\nAll session popularity charts saved to {output_dir}/")