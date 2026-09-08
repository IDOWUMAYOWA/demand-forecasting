import os
import sys
from pathlib import Path
import requests

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'app-ml', 'src'))
os.chdir(project_root)

# Force working directory to the project root
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from common.data_manager import DataManager
from common.utils import read_config, make_prediction_figures

# Load configuration using utils function
config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)

# Override host for Docker environment if environment variable is set
# inference_api_host = ...
# inference_api_port = ...
# inference_api_endpoint = ...
# INFERENCE_API_URL = f"http://{inference_api_host}:{inference_api_port}{inference_api_endpoint}"

# Use the default Bootstrap (light) theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the layout
app.layout = dbc.Container(
    [
    dbc.Row([
            ])
        ]
)


"""
Callbacks to update the User Interface

@callback(
    [Outputs],
    [Inputs] 
)
def update_graphs(Inputs):
    return Outputs
"""

server = app.server

if __name__ == '__main__':
    # Use debug=True for development, debug=False for production
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)