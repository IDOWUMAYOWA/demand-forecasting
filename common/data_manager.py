import pandas as pd
from typing import Dict, Any



class DataManager:
    """
    A utility class responsible for handling all data-related I/O operations
    and transformations used across the ML pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataManager with a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration parameters for paths and filenames.
        """
        self.config = config

    @staticmethod
    def append_data(data_to_append) -> pd.DataFrame:
        """
        Append new data to an existing DataFrame and reset the index.
        """
        return df_updated

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a parquet file.

        Args:
            path (str): Path to the parquet file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        return pd.read_parquet(path)

    @staticmethod
    def save_data(data: pd.DataFrame, path: str) -> None:
        """
        Save a DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): Data to be saved.
            path (str): Output file path.

        Returns:
            None
        """
        data.to_parquet(path, index=False)