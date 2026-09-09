import pandas as pd
from pathlib import Path
from typing import Dict, Any


class DataManager:
    """
    A utility class responsible for handling all data-related I/O operations
    and transformations used across the ML pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def append_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Append new incoming data to the existing production data history,
        persist the updated history back to disk, and return it.

        Args:
            new_data (pd.DataFrame): The newly arrived row(s) to append.

        Returns:
            pd.DataFrame: The full, updated historical dataset — this is what
                gets passed into the pipeline so lag features have real data
                to compute from, not just the new row in isolation.
        """
        prod_path = Path(self.config['data_manager']['prod_data_path'])

        if prod_path.exists():
            existing = self.load_data(str(prod_path))
            df_updated = pd.concat([existing, new_data], ignore_index=True)
        else:
            df_updated = new_data.copy()

        if 'datetime' in df_updated.columns:
            df_updated = df_updated.drop_duplicates(subset='datetime', keep='last')
            df_updated = df_updated.sort_values('datetime')

        df_updated = df_updated.reset_index(drop=True)

        self.save_data(df_updated, str(prod_path))

        return df_updated

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def save_data(data: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)