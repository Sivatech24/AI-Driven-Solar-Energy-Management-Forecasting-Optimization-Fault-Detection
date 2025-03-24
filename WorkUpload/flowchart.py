import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define box properties
box_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue")

# Define positions of the flowchart elements
positions = {
    "Data Collection": (5, 7),
    "Data Preprocessing": (5, 6),
    "AI Model Training": (5, 5),
    "Prediction & Optimization": (5, 4),
    "Dashboard & Monitoring": (5, 3)
}

# Draw the flowchart elements
for text, (x, y) in positions.items():
    ax.text(x, y, text, fontsize=10, ha="center", bbox=box_props)

# Draw arrows between elements
arrow_props = dict(arrowstyle="->", color="black", linewidth=1.5)
ax.annotate("", xy=positions["Data Preprocessing"], xytext=positions["Data Collection"], arrowprops=arrow_props)
ax.annotate("", xy=positions["AI Model Training"], xytext=positions["Data Preprocessing"], arrowprops=arrow_props)
ax.annotate("", xy=positions["Prediction & Optimization"], xytext=positions["AI Model Training"], arrowprops=arrow_props)
ax.annotate("", xy=positions["Dashboard & Monitoring"], xytext=positions["Prediction & Optimization"], arrowprops=arrow_props)

# Remove axes and display
ax.set_xlim(0, 10)
ax.set_ylim(2, 8)
ax.axis("off")
plt.title("Flowchart: AI-Driven Solar Energy System", fontsize=12, fontweight="bold")
plt.show()
