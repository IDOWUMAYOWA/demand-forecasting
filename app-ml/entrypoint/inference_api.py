import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from flask import Flask, jsonify
from common.utils import read_config
from common.data_manager import DataManager
from common.monitoring import DriftMonitor
from pipelines.pipeline_runner import PipelineRunner

app = Flask(__name__)


@app.route('/run-inference', methods=['POST'])
def run_inference():
    """
    Runs inference on the latest available data, logs the prediction for
    drift monitoring, and returns the result.
    """
    raw = data_manager.load_data(config['data_manager']['raw_data_path'])
    incoming = raw.iloc[[-1]]  # placeholder: same simulation as entrypoint/inference.py
    full_window = data_manager.append_data(incoming)

    df_pred = pipeline_runner.run_inference(df=full_window)
    prediction = df_pred.to_dict(orient='records')[0]

    monitor.log_prediction(
        predicted_at=prediction['predicted_at'],
        target_hour=prediction['target_hour'],
        predicted_cnt=prediction['predicted_cnt'],
    )

    return jsonify(prediction)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    data_manager = DataManager(config=config)
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)
    monitor = DriftMonitor(config=config)

    app.run(host="0.0.0.0", port=5001)