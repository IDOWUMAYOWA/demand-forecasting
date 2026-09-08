"""
Inference Pipeline Entrypoint:
- Loads configuration
- Runs inference pipeline
"""

import sys
from pathlib import Path
from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

if __name__ == "__main__":

    # Load config file
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    # Initialize Pipeline Runner
    pipeline_runner = PipelineRunner(config=config)

    """
    Load data for inference
    df = ....
    """
    pipeline_runner.run_inference(df)