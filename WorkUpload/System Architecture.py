import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define nodes with states and descriptions
nodes = {
    "State 1": "Data Collection\n(Solar Panels, Weather Sensors, Inverter Logs)",
    "State 2": "Data Preprocessing\n(Cleaning, Normalization, Feature Engineering)",
    "State 3": "AI Model Training\n(LSTM, XGBoost, CNN-LSTM Hybrid)",
    "State 4": "Prediction & Optimization\n(Energy Forecasting, Failure Detection)",
    "State 5": "Dashboard & Monitoring\n(Streamlit Visualization, Real-time Monitoring)"
}

# Add nodes to the graph
G.add_nodes_from(nodes.keys())

# Define edges (connections between states)
edges = [("State 1", "State 2"), ("State 2", "State 3"), ("State 3", "State 4"), ("State 4", "State 5")]
G.add_edges_from(edges)

# Define hierarchical positions with increased spacing to avoid overlaps
pos = {
    "State 1": (0, 8),
    "State 2": (0, 6),
    "State 3": (0, 4),
    "State 4": (0, 2),
    "State 5": (0, 0)
}

# Draw the graph with adjusted spacing and font size
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=False, node_color="lightblue", edge_color="black",
        node_size=4500, font_size=10, font_weight="bold", arrows=True, arrowsize=20)

# Add labels with better spacing and background for clarity
for state, description in nodes.items():
    x, y = pos[state]
    plt.text(x, y - 0.5, description, fontsize=10, ha='center', fontweight="bold", 
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

# Display the improved architecture diagram
plt.title("AI-Driven Solar Energy System Architecture", fontsize=14, fontweight="bold")
plt.show()
