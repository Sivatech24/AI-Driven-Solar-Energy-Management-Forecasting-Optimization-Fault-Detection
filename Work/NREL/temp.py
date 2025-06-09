import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt
import time

# === Load data once ===
df = pd.read_csv("wy-pv-2006.csv")
df['local_time'] = pd.to_datetime(df['local_time'], format="%m/%d/%y %H:%M")

# === Plot setup ===
fig, ax = plt.subplots()
times = []
powers = []
line, = ax.plot_date(times, powers, linestyle='solid', marker=None)

plt.title("Real-Time Solar Power Output")
plt.xlabel("Local Time")
plt.ylabel("Power (MW)")
plt.xticks(rotation=45)
plt.tight_layout()

# === Animation update function ===
index = [0]  # mutable holder to keep track of index across calls

def update(frame):
    if index[0] >= len(df):
        return line,

    # Append new data
    row = df.iloc[index[0]]
    times.append(row['local_time'])
    powers.append(row['power_mw'])
    index[0] += 1

    # Update line
    line.set_data(times, powers)

    # Rescale X/Y limits dynamically
    ax.relim()
    ax.autoscale_view()

    return line,

# === Animate ===
ani = animation.FuncAnimation(fig, update, interval=1000)  # 1000ms = 1 second
plt.show()
