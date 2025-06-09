import os
import re
import pandas as pd
from tqdm import tqdm

def parse_filename(filename):
    match = re.match(r'(Actual|DA|HA4)_(.+)_(\d{4})_(DPV|UPV)_(\d+)MW_(\d+_Min)\.csv', filename)
    if match:
        _, coords, year, site_type, capacity, resolution = match.groups()
        lat, lon = map(float, coords.split('_'))
        return lat, lon, int(year), site_type, int(capacity), resolution
    return None

base_path = r"C:\Users\tech\Downloads\NREL\ProcessedData\fl-pv-2006"
output_csv_path = "combined_fl-pv-2006.csv"

csv_files = []
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append((root, file))

columns = [
    "latitude", "longitude", "year", "site_type", "capacity_mw",
    "resolution", "row_index", "local_time", "power_mw"
]

# Remove existing output file if exists
if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

with tqdm(total=len(csv_files), desc="Combining CSV files", unit="file") as pbar:
    for root, file in csv_files:
        parsed = parse_filename(file)
        if not parsed:
            pbar.write(f"⚠️ Skipping invalid filename: {file}")
            pbar.update(1)
            continue

        lat, lon, year, site_type, capacity, resolution = parsed
        file_path = os.path.join(root, file)

        try:
            df = pd.read_csv(file_path)
            if 'LocalTime' not in df.columns or 'Power(MW)' not in df.columns:
                pbar.write(f"⚠️ Skipping file with missing columns: {file}")
                pbar.update(1)
                continue
        except Exception as e:
            pbar.write(f"❌ Error reading {file}: {e}")
            pbar.update(1)
            continue

        # Add metadata columns
        df_out = pd.DataFrame({
            "latitude": lat,
            "longitude": lon,
            "year": year,
            "site_type": site_type,
            "capacity_mw": capacity,
            "resolution": resolution,
            "row_index": range(len(df)),
            "local_time": df["LocalTime"],
            "power_mw": df["Power(MW)"]
        })

        # Write header only for first file
        if pbar.n == 0:
            df_out.to_csv(output_csv_path, mode='w', index=False)
        else:
            df_out.to_csv(output_csv_path, mode='a', header=False, index=False)

        pbar.update(1)

print(f"\n✅ Combined CSV saved as '{output_csv_path}'")
