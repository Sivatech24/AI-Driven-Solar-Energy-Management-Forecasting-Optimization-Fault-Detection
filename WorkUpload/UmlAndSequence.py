from graphviz import Digraph

# Creating UML Entity-Relationship Diagram (ERD)
erd = Digraph("ERD", format="png")
erd.attr(rankdir="LR", size="10")

# Entities
entities = {
    "SolarPanel": ["id", "location", "capacity", "installation_date"],
    "SensorData": ["id", "solar_panel_id", "temperature", "irradiance", "voltage", "current", "timestamp"],
    "PowerOutput": ["id", "solar_panel_id", "generated_power", "efficiency", "timestamp"],
    "Inverter": ["id", "model", "status", "solar_panel_id"],
    "FailureLogs": ["id", "inverter_id", "error_code", "timestamp", "resolved"]
}

# Adding nodes
for entity, attributes in entities.items():
    erd.node(entity, label=f"{entity}\n" + "\n".join(attributes), shape="box")

# Relationships
erd.edge("SolarPanel", "SensorData", label="records")
erd.edge("SolarPanel", "PowerOutput", label="produces")
erd.edge("SolarPanel", "Inverter", label="connected to")
erd.edge("Inverter", "FailureLogs", label="logs failures")

# Save ERD Diagram
erd_file_path = "/mnt/data/solar_erd"
erd.render(erd_file_path)

# Creating System Sequence Diagram
seq = Digraph("SequenceDiagram", format="png")
seq.attr(rankdir="TB", size="10")

# Components
components = ["User", "SolarPanelSystem", "ML Model", "Database"]

# Adding components as nodes
for comp in components:
    seq.node(comp, shape="rectangle", style="filled", fillcolor="lightblue")

# Interactions (sequence flow)
seq.edge("User", "SolarPanelSystem", label="Request Data")
seq.edge("SolarPanelSystem", "Database", label="Fetch Sensor Data")
seq.edge("SolarPanelSystem", "ML Model", label="Analyze Data")
seq.edge("ML Model", "SolarPanelSystem", label="Send Predictions")
seq.edge("SolarPanelSystem", "User", label="Show Results")

# Save Sequence Diagram
seq_file_path = "/mnt/data/solar_sequence"
seq.render(seq_file_path)

erd_file_path + ".png", seq_file_path + ".png"
