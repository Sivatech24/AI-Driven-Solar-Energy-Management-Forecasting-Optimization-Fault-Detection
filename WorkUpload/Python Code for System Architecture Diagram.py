import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define system components (nodes)
nodes = [
    "Data Collection", 
    "Data Preprocessing", 
    "Machine Learning Models", 
    "Prediction & Optimization", 
    "Dashboard & Monitoring"
]

# Add nodes to the graph
G.add_nodes_from(nodes)

# Define edges (connections)
edges = [
    ("Data Collection", "Data Preprocessing"),
    ("Data Preprocessing", "Machine Learning Models"),
    ("Machine Learning Models", "Prediction & Optimization"),
    ("Prediction & Optimization", "Dashboard & Monitoring")
]

# Add edges to the graph
G.add_edges_from(edges)

# Define node positions for hierarchical layout
pos = {
    "Data Collection": (0, 4),
    "Data Preprocessing": (0, 3),
    "Machine Learning Models": (0, 2),
    "Prediction & Optimization": (0, 1),
    "Dashboard & Monitoring": (0, 0)
}

# Draw the network graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=3000, font_size=10, font_weight="bold", arrows=True, arrowsize=20)

# Display the architecture diagram
plt.title("AI-Driven Solar Energy System Architecture", fontsize=12, fontweight="bold")
plt.show()
