import pandas as pd
import numpy as np
import os

# --- Load processed data ---
df = pd.read_csv("../data/final/cleaned_player_data.csv")


# --- Clean and prepare ---
# --- Detect and convert the date column safely ---
possible_date_cols = [c for c in df.columns if "date" in c.lower()]
if possible_date_cols:
    date_col = possible_date_cols[0]
    print("🧩 Columns in dataset:", df.columns.tolist())

    df["match_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
else:
    print("⚠️ No date column found; creating synthetic dates.")
    df["match_date"] = pd.date_range(start="2015-01-01", periods=len(df), freq="7D")

df = df.sort_values(['player_id', 'match_date']).reset_index(drop=True)


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
    df.groupby('player_id')['match_date']

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

# ============================================================
# 💡 NEW SECTION: Extra engineered features for ML model
# ============================================================

# --- Rolling minutes played (simulate using rating_rolling as proxy if minutes not in data) ---
if "minutes_played" in df.columns:
    df['rolling_minutes_3m'] = (
        df.groupby("player_id")["minutes_played"]
          .rolling(window=3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
else:
    df['rolling_minutes_3m'] = (
        df.groupby("player_id")["rating_rolling"]
          .rolling(window=3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True) * 0.9
    )

# --- Pressures (approximate: higher sprint_norm & lower recovery => more pressure events) ---
df['pressures'] = ((1 - df['recovery_factor']) * df['sprint_norm'] * 100).clip(0, 100)

# --- Sprints (approximate: correlated with sprint speed and fatigue) ---
df['sprints'] = (df['sprint_speed'] * df['fatigue_index'] / 2).clip(0, 100)

# --- Days gap since last match ---
df['previous_match_gap_days'] = df['days_since_last_match']

# ============================================================
# ✅ Save final dataset for model & dashboard
# ============================================================

os.makedirs("data/final", exist_ok=True)
output_path = "data/final/cleaned_player_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Final dataset saved to {output_path}")
print("✅ Columns available for ML model:")
print(["fatigue_index", "rolling_minutes_3m", "pressures", "sprints", "previous_match_gap_days"])

# --- Quick validation ---
print("\n📊 Feature Summary:")
print(df[["fatigue_index", "rolling_minutes_3m", "pressures", "sprints", "previous_match_gap_days"]]
      .describe()
      .round(2)
      .to_string())
