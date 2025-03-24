from graphviz import Digraph
import os

# Create a new sequence diagram for Solar Panel Analysis AI Project
solar_seq_diag = Digraph('Solar Panel Analysis Sequence Diagram', format='png')

# Define entities in the sequence
solar_seq_diag.node('U', 'User')
solar_seq_diag.node('DC', 'Solar Data Collection')
solar_seq_diag.node('DP', 'Data Preprocessing')
solar_seq_diag.node('ML1', 'LSTM Model (Power Prediction)')
solar_seq_diag.node('ML2', 'Random Forest (Failure Detection)')
solar_seq_diag.node('PO', 'Prediction & Optimization')
solar_seq_diag.node('DB', 'Streamlit Dashboard')

# Define interactions (sequence flow)
solar_seq_diag.edge('U', 'DC', 'Request Solar Data')
solar_seq_diag.edge('DC', 'DP', 'Send Raw Data')
solar_seq_diag.edge('DP', 'ML1', 'Preprocessed Data for Power Prediction')
solar_seq_diag.edge('DP', 'ML2', 'Preprocessed Data for Failure Detection')
solar_seq_diag.edge('ML1', 'PO', 'Solar Power Forecasts')
solar_seq_diag.edge('ML2', 'PO', 'Failure Prediction Results')
solar_seq_diag.edge('PO', 'DB', 'Optimized Energy & Alerts')
solar_seq_diag.edge('DB', 'U', 'Display Forecast & System Status')

# Render and display the sequence diagram
solar_seq_diag.attr(size='10')
solar_seq_diag

# Define a new file path to save the UML diagram
uml_file_path = "C:/Users/tech/Documents/solar_panel_analysis_sequence"
solar_seq_diag.render(uml_file_path, format="png")

# Render and save the sequence diagram as a PNG file
solar_seq_diag.render(uml_file_path, format="png")

# Provide the correct file path for download
uml_file_path + ".png"
