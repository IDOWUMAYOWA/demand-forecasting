import os
import sys
from pathlib import Path
import requests

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'app-ml', 'src'))
os.chdir(project_root)

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from common.data_manager import DataManager
from common.utils import read_config, make_prediction_figures

config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)
data_manager = DataManager(config=config)

# Override host/port via env vars in Docker (compose service name resolves
# via internal DNS; 'localhost' only works when running app-ui outside Docker)
inference_api_host = os.environ.get('INFERENCE_API_HOST', config['inference_api']['host'])
inference_api_port = os.environ.get('INFERENCE_API_PORT', config['inference_api']['port'])
inference_api_endpoint = config['inference_api']['endpoint']
INFERENCE_API_URL = f"http://{inference_api_host}:{inference_api_port}{inference_api_endpoint}"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row([dbc.Col(html.H2("Bike Demand Forecast"), width=12)], className="mt-3"),
        dbc.Row(
            [
                dbc.Col(dbc.Button("Get Next-Hour Prediction", id="predict-button", color="primary"), width=3),
                dbc.Col(html.Div(id="prediction-summary"), width=9),
            ],
            className="mb-4",
        ),
        dbc.Row([dbc.Col(dcc.Graph(id="prediction-graph"), width=12)]),
    ]
)


@callback(
    [Output("prediction-graph", "figure"), Output("prediction-summary", "children")],
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_graphs(n_clicks):
    response = requests.post(INFERENCE_API_URL)
    response.raise_for_status()
    prediction = response.json()

    history = data_manager.load_data(config['data_manager']['raw_data_path'])
    recent = history.tail(config['ui']['history_hours'])

    figure = make_prediction_figures(recent, prediction)
    summary = html.Div([
        html.Strong(f"Predicted count for {prediction['target_hour']}: "),
        html.Span(f"{prediction['predicted_cnt']}"),
    ])
    return figure, summary


server = app.server

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)