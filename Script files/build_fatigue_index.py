import pandas as pd
import numpy as np
import os

# --- Load processed data ---
df = pd.read_csv("data/processed/player_attributes.csv")

# --- Clean and prepare ---
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['player_id', 'date']).reset_index(drop=True)

# Fill missing numeric values per player
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df.groupby('player_id')[num_cols].ffill()

print("✅ Data cleaned and sorted!")
print(df.head())

# Rolling average of overall_rating (last 3 matches)
df['rating_rolling'] = (
    df.groupby('player_id')['overall_rating']
      .rolling(window=3, min_periods=1)
      .mean()
      .reset_index(0, drop=True)
)

# Days between matches (to measure recovery)
df['days_since_last_match'] = (
    df.groupby('player_id')['date']
      .diff()
      .dt.days
)
df['days_since_last_match'] = df['days_since_last_match'].fillna(7)  # assume 7 days rest initially

# --- Normalize attributes ---
df['stamina_norm'] = df['stamina'].clip(0, 100) / 100.0
df['sprint_norm']  = df['sprint_speed'].clip(0, 100) / 100.0
days = df['days_since_last_match'].astype(float)

# --- Recovery Model ---
rec_base = 1.0 - np.exp(- days / 7.0)   # 7-day recovery curve
penalty_strength = 0.12
penalty_threshold = 30.0
penalty_scale = 90.0
penalty = penalty_strength * (1.0 - np.exp(-(np.maximum(0.0, days - penalty_threshold) / penalty_scale)))

# --- Add sharpness decay (loss of form with long inactivity) ---
df['sharpness_decay'] = np.exp(-((days - 30)**2) / (2 * 40**2))  # peak at ~30 days rest
df['recovery_factor'] = (rec_base * (1 - penalty) * df['sharpness_decay']).clip(0.0, 1.0)

# --- Fatigue Index ---
df['fatigue_index'] = (
    0.5 * (1 - df['stamina_norm']) +
    0.3 * (df['sprint_norm'] * (1 - df['recovery_factor'])) +
    0.2 * (1 - df['recovery_factor'])
)

# --- Save final dataset ---
os.makedirs("data/final", exist_ok=True)
df.to_csv("data/final/player_fatigue_dataset.csv", index=False)
print("\n✅ Final dataset with Fatigue Index saved to data/final/player_fatigue_dataset.csv")

# --- Final sanity checks & summaries ---
print("\n🧠 Sanity-check (days -> recovery_factor):")
sample_days = pd.DataFrame({'days_since_last_match':[2,7,21,60,189]})
sample_days['rec_base'] = 1 - np.exp(- sample_days['days_since_last_match'] / 7.0)
penalty_strength = 0.12
penalty_threshold = 30.0
penalty_scale = 90.0
sample_days['penalty'] = penalty_strength * (1 - np.exp(-(np.maximum(0.0, sample_days['days_since_last_match'] - penalty_threshold) / penalty_scale)))
sample_days['recovery_factor'] = (sample_days['rec_base'] - sample_days['penalty']).clip(0,1)
print(sample_days.to_string(index=False))

# --- Sample player fatigue data ---
print("\n🔍 Sample player fatigue data:")
print(df[['player_name', 'date', 'stamina', 'sprint_speed', 'days_since_last_match', 'fatigue_index']]
      .head(10)
      .to_string(index=False))

# --- Top 10 Most Fatigued Players ---
top_fatigued = df.nlargest(10, 'fatigue_index')[['player_name', 'fatigue_index', 'stamina', 'sprint_speed', 'days_since_last_match']]
print("\n🔥 Top 10 Most Fatigued Players:")
print(top_fatigued.to_string(index=False))

# --- Top 10 Freshest Players ---
top_fresh = df.nsmallest(10, 'fatigue_index')[['player_name', 'fatigue_index', 'stamina', 'sprint_speed', 'days_since_last_match']]
print("\n💧 Top 10 Freshest Players:")
print(top_fresh.to_string(index=False))

# --- Missing Values Check ---
print("\n🧹 Missing Values Check:")
print(df.isnull().sum().to_string())

# --- Data Summary ---
print("\n📊 Data Summary (key metrics):")
print(df[['fatigue_index', 'stamina', 'sprint_speed', 'days_since_last_match']].describe().round(2).to_string())


