import os
import re
import time
import pandas as pd
import sqlite3
from tqdm import tqdm

def parse_filename(filename):
    match = re.match(r'(Actual|DA|HA4)_(.+)_(\d{4})_(DPV|UPV)_(\d+)MW_(\d+_Min)\.csv', filename)
    if match:
        _, coords, year, site_type, capacity, resolution = match.groups()
        lat, lon = map(float, coords.split('_'))
        return lat, lon, int(year), site_type, int(capacity), resolution
    return None

# Path to root folder
base_path = r"C:/Users/tech/Downloads/NREL/ProcessedData/wy-pv-2006"

# SQLite setup
conn = sqlite3.connect("solar_data.db")
cursor = conn.cursor()

# Updated schema without filename, folder_path, data_type
cursor.execute('''
    CREATE TABLE IF NOT EXISTS solar_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL,
        longitude REAL,
        year INTEGER,
        site_type TEXT,
        capacity_mw INTEGER,
        resolution TEXT,
        row_index INTEGER,
        local_time TEXT,
        power_mw REAL
    )
''')

# Gather all .csv files recursively
csv_files = []
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append((root, file))

start_time = time.time()

batch_size = 10000
batch_records = []
with tqdm(total=len(csv_files), desc="Processing CSV files", unit="file") as pbar:
    for root, file in csv_files:
        parsed = parse_filename(file)
        if not parsed:
            pbar.write(f"Skipping invalid filename: {file}")
            pbar.update(1)
            continue

        lat, lon, year, site_type, capacity, resolution = parsed
        file_path = os.path.join(root, file)

        try:
            df = pd.read_csv(file_path)
            if 'LocalTime' not in df.columns or 'Power(MW)' not in df.columns:
                pbar.write(f"Skipping file with missing columns: {file}")
                pbar.update(1)
                continue
        except Exception as e:
            pbar.write(f"Error reading {file}: {e}")
            pbar.update(1)
            continue

        for idx, row in df.iterrows():
            batch_records.append((
                lat, lon, year, site_type, capacity, resolution,
                int(idx), row['LocalTime'], row['Power(MW)']
            ))

        if len(batch_records) >= batch_size:
            cursor.executemany('''
                INSERT INTO solar_data (
                    latitude, longitude, year, site_type, capacity_mw,
                    resolution, row_index, local_time, power_mw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch_records)
            conn.commit()
            batch_records.clear()

        pbar.update(1)

    if batch_records:
        cursor.executemany('''
            INSERT INTO solar_data (
                latitude, longitude, year, site_type, capacity_mw,
                resolution, row_index, local_time, power_mw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_records)
        conn.commit()

conn.close()
total_time = time.time() - start_time
print(f"\n✅ Done! solar_data.db created in {int(total_time)} seconds.")
