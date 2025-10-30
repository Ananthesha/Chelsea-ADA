import sqlite3
import pandas as pd

# Path to your SQLite file
db_path = "/Users/msananthesha/Desktop/ada project/database.sqlite"


# Connect to the database
conn = sqlite3.connect(db_path)

# List all available tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Available tables:")
print(tables)

# Example: read the Player table
players = pd.read_sql_query("SELECT * FROM Player LIMIT 5;", conn)
print("\nSample from Player table:")
print(players)

# Example: read the Match table
matches = pd.read_sql_query("SELECT * FROM Match LIMIT 5;", conn)
print("\nSample from Match table:")
print(matches.columns)

conn.close()
