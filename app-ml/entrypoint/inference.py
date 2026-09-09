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

    # In production this row comes from a real upstream source
    # (an API call, a scheduled data drop, etc.) — for now, simulate
    # by reading the most recent row from raw data.
    raw = data_manager.load_data(config['data_manager']['raw_data_path'])
    incoming = raw.iloc[[-1]]

    full_window = data_manager.append_data(incoming)

    runner = PipelineRunner(config=config, data_manager=data_manager)
    df_pred = runner.run_inference(df=full_window)

    print(df_pred)
    return df_pred


if __name__ == '__main__':
    main()