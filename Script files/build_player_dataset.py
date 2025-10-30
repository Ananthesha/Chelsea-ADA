import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("database.sqlite")

# Query player attributes and join with player names
query = """
SELECT 
    pa.player_api_id AS player_id,
    p.player_name,
    pa.date,
    pa.overall_rating,
    pa.potential,
    pa.stamina,
    pa.strength,
    pa.sprint_speed,
    pa.agility,
    pa.reactions
FROM Player_Attributes pa
JOIN Player p ON pa.player_api_id = p.player_api_id
"""
df = pd.read_sql_query(query, conn)

conn.close()

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by player and date
df = df.sort_values(['player_id', 'date']).reset_index(drop=True)

print("✅ Dataset built successfully!")
print(df.head())
print("\nShape:", df.shape)

# Save for later steps
import os

# Make sure the folder exists before saving
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/player_attributes.csv", index=False)
print("✅ Saved processed data to data/processed/player_attributes.csv")

df.to_csv("data/processed/player_attributes.csv", index=False)
