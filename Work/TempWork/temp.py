import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.colors as colors

# === Load data ===
df = pd.read_csv("wy-pv-2006.csv")
df['local_time'] = pd.to_datetime(df['local_time'], format="%m/%d/%y %H:%M")

lat = df['latitude'].iloc[0]
lon = df['longitude'].iloc[0]

# Normalize power for color and size
norm = colors.Normalize(vmin=df['power_mw'].min(), vmax=df['power_mw'].max())
cmap = cm.get_cmap('YlOrRd')

# === Setup plots ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Time Series Plot ---
times = []
powers = []
line, = ax1.plot_date(times, powers, linestyle='solid', marker=None)
ax1.set_title("Real-Time Solar Power Output")
ax1.set_xlabel("Local Time")
ax1.set_ylabel("Power (MW)")
ax1.tick_params(axis='x', rotation=45)

# --- Terrain Map with bluemarble ---
m = Basemap(projection='merc',
            llcrnrlat=lat - 5, urcrnrlat=lat + 5,
            llcrnrlon=lon - 5, urcrnrlon=lon + 5,
            resolution='i', ax=ax2)

m.bluemarble()  # <-- Apply blue marble terrain

m.drawcountries()
m.drawstates()
m.drawrivers()
m.drawparallels(range(int(lat)-5, int(lat)+6, 2), labels=[1,0,0,0])
m.drawmeridians(range(int(lon)-5, int(lon)+6, 2), labels=[0,0,0,1])
ax2.set_title("Satellite Terrain Map (BlueMarble) with Solar Output Visualization")

# Convert lat/lon to map projection coordinates
x, y = m(lon, lat)

# Colored area for energy production
solar_patch = m.scatter(x, y, s=0, c='gray', alpha=0.5, edgecolors='none', zorder=5)

# === Animation Update ===
index = [0]
def update(frame):
    if index[0] >= len(df):
        return line, solar_patch

    row = df.iloc[index[0]]
    time = row['local_time']
    power = row['power_mw']

    # Line chart update
    times.append(time)
    powers.append(power)
    line.set_data(times, powers)
    ax1.relim()
    ax1.autoscale_view()

    # Map energy area update
    color = cmap(norm(power))
    size = 300 + power * 500
    alpha = 0.4 + 0.5 * norm(power)

    solar_patch.set_offsets([[x, y]])
    solar_patch.set_sizes([size])
    solar_patch.set_color([color])
    solar_patch.set_alpha(alpha)

    index[0] += 1
    return line, solar_patch

ani = animation.FuncAnimation(fig, update, interval=500)
plt.tight_layout()
plt.show()
