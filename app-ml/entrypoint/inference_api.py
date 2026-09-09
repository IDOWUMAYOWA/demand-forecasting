import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from flask import Flask, jsonify
from common.utils import read_config
from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner

app = Flask(__name__)


@app.route('/run-inference', methods=['POST'])
def run_inference():
    """
    Runs inference on the latest available data and returns the prediction.
    """
    raw = data_manager.load_data(config['data_manager']['raw_data_path'])
    incoming = raw.iloc[[-1]]  # placeholder: same simulation as entrypoint/inference.py
    full_window = data_manager.append_data(incoming)

    df_pred = pipeline_runner.run_inference(df=full_window)
    return jsonify(df_pred.to_dict(orient='records')[0])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    data_manager = DataManager(config=config)
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)

    app.run(host="0.0.0.0", port=5001)