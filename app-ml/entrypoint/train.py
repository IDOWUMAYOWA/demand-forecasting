"""
Training Pipeline Entrypoint
- Loads configuration
- Runs the full training pipeline (preprocessing, feature engineering, training, postprocessing)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner


if __name__ == "__main__":

    # Load config file
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    """
    Load data to train am ML model
    df = ...
    """

    # Initialize Pipeline Runner
    pipeline_runner = PipelineRunner(config=config)

    # Run the training pipeline
    pipeline_runner.run_training(df)