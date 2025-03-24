import matplotlib.pyplot as plt
import networkx as nx

# Create a graph for the ERD
G = nx.DiGraph()

# Define entities and their attributes
entities = {
    "PlantData": ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"],
    "WeatherData": ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
    "Plant": ["PLANT_ID", "PLANT_NAME", "LOCATION"],
    "Source": ["SOURCE_KEY", "SOURCE_TYPE"]
}

# Add nodes for entities
for entity, attributes in entities.items():
    G.add_node(entity, label=f"{entity}\n" + "\n".join(attributes))

# Define relationships (edges)
relationships = [
    ("PlantData", "Plant"),
    ("PlantData", "Source"),
    ("WeatherData", "Plant"),
    ("WeatherData", "Source"),
]

# Add edges for relationships
G.add_edges_from(relationships)

# Plot the ERD
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
nx.draw(G, pos, with_labels=True, node_size=5000, node_color="lightblue", edge_color="gray", font_size=9, font_weight="bold")

# Save the diagram
file_path = "solar_plant_erd.png"
plt.savefig(file_path, format="png")
plt.close()

# Return the file path
file_path
