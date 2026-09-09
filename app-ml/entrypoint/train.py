import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from common.utils import read_config
from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner


def main():
    config = read_config('config/config.yaml')
    data_manager = DataManager(config=config)

    df = data_manager.load_data(config['data_manager']['raw_data_path'])

    runner = PipelineRunner(config=config, data_manager=data_manager)
    runner.run_training(df=df)

    print(f"Training complete. Model saved to {config['data_manager']['model_path']}")


if __name__ == '__main__':
    main()