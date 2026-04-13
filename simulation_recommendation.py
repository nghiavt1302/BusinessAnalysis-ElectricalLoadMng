import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Base config
FOLDER_PATH = 'F:/study/output/'
FILE_LIST = glob.glob(os.path.join(FOLDER_PATH, "LCL-June2015v2_*.csv"))
PEAK_HOURS = [16, 17, 18, 19, 20, 21]
BASE_TEMP = 24.0 # Base Temp, for calculating CDD


print(f"Start processing {len(FILE_LIST)} files...")

# Temporary data lists
sys_load_chunks = []
cust_basic_chunks = []
weather_chunks = []
hourly_profile_chunks = []

# Map, grouping each files
for i, file in enumerate(FILE_LIST):
    print(f"Processing [{i+1}/{len(FILE_LIST)}]: {os.path.basename(file)}")
    df = pd.read_csv(file, parse_dates=['timestamp'])
    
    # Time columns
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    df['is_peak'] = df['hour'].isin(PEAK_HOURS)
    
    # Simulate 5%, 10%
    df['kwh_sim_5pct'] = np.where(df['is_peak'], df['kwh'] * 0.95, df['kwh'])
    df['kwh_sim_10pct'] = np.where(df['is_peak'], df['kwh'] * 0.90, df['kwh'])
    
    sys_chunk = df.groupby(['date', 'hour']).agg({
        'kwh': 'sum', 'kwh_sim_5pct': 'sum', 'kwh_sim_10pct': 'sum', 'is_peak': 'first'
    }).reset_index()
    sys_load_chunks.append(sys_chunk)
    
    # 2. Customer Basic (Rule-based)
    df['peak_kwh'] = np.where(df['is_peak'], df['kwh'], 0)
    cust_chunk = df.groupby('meter_id').agg({'kwh': 'sum', 'peak_kwh': 'sum'}).reset_index()
    cust_basic_chunks.append(cust_chunk)
    
    # 3. Weather & CDD
    weather_chunk = df.groupby(['meter_id', 'date']).agg({'kwh': 'sum', 'temp': 'mean'}).reset_index()
    weather_chunks.append(weather_chunk)
    
    # 4. Load Profiling (K-Means)
    hr_chunk = df.groupby(['meter_id', 'hour']).agg(
        kwh_sum=('kwh', 'sum'),
        kwh_count=('kwh', 'count')
    ).reset_index()
    hourly_profile_chunks.append(hr_chunk)

print("\nDone! Reducing data...")


# EXPORT FILE
# System Simulation & Peak Summary
final_sys = pd.concat(sys_load_chunks).groupby(['date', 'hour']).sum().reset_index()
final_sys['is_peak'] = final_sys['is_peak'] > 0 
final_sys.to_csv('1_system_simulation.csv', index=False)

daily_peak = final_sys[final_sys['is_peak']].groupby('date').agg({
    'kwh': 'max', 'kwh_sim_5pct': 'max', 'kwh_sim_10pct': 'max'
}).reset_index()
daily_peak['reduce_5pct'] = daily_peak['kwh'] - daily_peak['kwh_sim_5pct']
daily_peak['reduce_10pct'] = daily_peak['kwh'] - daily_peak['kwh_sim_10pct']
daily_peak.to_csv('2_peak_reduction_summary.csv', index=False)

# Weather & CDD
final_weather = pd.concat(weather_chunks).groupby(['meter_id', 'date']).mean().reset_index()
final_weather['CDD'] = np.maximum(final_weather['temp'] - BASE_TEMP, 0)

# Base load
base_load_df = final_weather[final_weather['CDD'] == 0].groupby('meter_id')['kwh'].mean().reset_index(name='base_load')
final_weather = final_weather.merge(base_load_df, on='meter_id', how='left')
final_weather['base_load'] = final_weather['base_load'].fillna(final_weather.groupby('meter_id')['kwh'].transform('min'))
final_weather['cooling_load'] = np.maximum(final_weather['kwh'] - final_weather['base_load'], 0)

# Group by day
system_weather = final_weather.groupby('date').agg({
    'kwh': 'sum', 'base_load': 'sum', 'cooling_load': 'sum', 'temp': 'mean'
}).reset_index()
system_weather.to_csv('3_weather_impact_daily.csv', index=False)

# Total Cooling Load for each customer
cust_cooling = final_weather.groupby('meter_id').agg({'cooling_load': 'sum', 'base_load': 'sum'}).reset_index()

# K-means
final_hr = pd.concat(hourly_profile_chunks).groupby(['meter_id', 'hour']).sum().reset_index()
final_hr['avg_kwh'] = final_hr['kwh_sum'] / final_hr['kwh_count']

pivot_df = final_hr.pivot(index='meter_id', columns='hour', values='avg_kwh').fillna(0)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(pivot_df.T).T 

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
pivot_df['cluster_id'] = kmeans.fit_predict(normalized_data)

def name_cluster(row):
    peak_h = row.drop('cluster_id').idxmax()
    if 0 <= peak_h <= 6: return "Night Owl"
    elif 7 <= peak_h <= 16: return "Daytime Active"
    else: return "Evening Peak"

pivot_df['cluster_name'] = pivot_df.apply(name_cluster, axis=1)

# Average shape of each cluster
cluster_shapes = pivot_df.groupby('cluster_name').mean().drop(columns=['cluster_id']).T.reset_index()
cluster_shapes.melt(id_vars='hour', var_name='cluster_name', value_name='avg_kwh').to_csv('4_kmeans_hourly_shapes.csv', index=False)

# Customer Profiles & Rule-Based
final_cust = pd.concat(cust_basic_chunks).groupby('meter_id').sum().reset_index()
final_cust.rename(columns={'kwh': 'total_kwh'}, inplace=True)
final_cust['peak_ratio'] = np.where(final_cust['total_kwh']>0, final_cust['peak_kwh']/final_cust['total_kwh'], 0)

# Merge with Clustering & Cooling Load
final_cust = final_cust.merge(pivot_df[['cluster_name']].reset_index(), on='meter_id', how='left')
final_cust = final_cust.merge(cust_cooling, on='meter_id', how='left')

# Rule-based
q_kwh = final_cust['total_kwh'].quantile(0.75)
q_peak = 0.35
def assign_level(row):
    if row['total_kwh'] > q_kwh and row['peak_ratio'] > q_peak: return 'Level 3 - Demand Response'
    elif row['peak_ratio'] > q_peak: return 'Level 2 - TOU Pricing'
    else: return 'Level 1 - Communication'

final_cust['policy_level'] = final_cust.apply(assign_level, axis=1)
final_cust.to_csv('5_customer_profiles_advanced.csv', index=False)

print("\nDone!")