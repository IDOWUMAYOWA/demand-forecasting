import sys
from pathlib import Path
from flask import Flask, jsonify
from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

app = Flask(__name__)

@app.route('/run-inference', methods=['POST'])
def run_inference():
    """
    Load the dataset to run inference on
    df = ....
    inference_timestamp = .....
    df = df.iloc[inference_timestamp, :]
    """
    pipeline_runner.run_inference(df)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Load configuration using utils function
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    # Initialize Pipeline Runner
    pipeline_runner = PipelineRunner(config)

    # Start the app
    app.run(host="0.0.0.0", port=5001) 