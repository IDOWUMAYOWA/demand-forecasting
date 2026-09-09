import yaml
from pathlib import Path
from typing import Union, Dict, Any
import pandas as pd



def read_config(path: Union[str, Path]) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def make_prediction_figures(history: pd.DataFrame, prediction: Dict[str, Any]) -> "go.Figure":
    """
    Build a line chart of recent actual demand, with the new next-hour
    prediction plotted as a distinct marker right after it.

    Args:
        history (pd.DataFrame): Recent rows with a 'cnt' column.
        prediction (dict): The JSON response from the inference API —
            expects 'predicted_cnt' and 'target_hour'.

    Returns:
        go.Figure: A plotly figure ready to hand to dcc.Graph.
    
    """
    import plotly.graph_objects as go
    x_hist = list(range(len(history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=history['cnt'],
        mode='lines+markers',
        name='Actual (recent)',
    ))
    fig.add_trace(go.Scatter(
        x=[len(history)],
        y=[prediction['predicted_cnt']],
        mode='markers',
        marker=dict(size=14, color='red', symbol='star'),
        name='Predicted (next hour)',
    ))
    fig.update_layout(
        title='Recent Demand and Next-Hour Forecast',
        xaxis_title='Hour (relative)',
        yaxis_title='Bike Count',
        template='plotly_white',
    )
    return fig