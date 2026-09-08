import pandas as pd
from typing import Dict


class PreprocessingPipeline:
    """
    A pipeline for preprocessing the raw data.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config

    def run(self, df: pd.DataFrame):
        """
        Execute the complete preprocessing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be preprocessed

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # return df