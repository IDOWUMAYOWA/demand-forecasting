import pandas as pd
from typing import Dict


class PreprocessingPipeline:
    """
    A pipeline for preprocessing the raw data.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.drop_cols = config['preprocessing']['drop_cols']

    def run(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be preprocessed
            is_train (bool): If True, creates the shifted target column
                (used for training). If False, skips it (used for inference,
                where the next-hour value is unknown).

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df = df.copy()

        if is_train:
            df['target'] = df['cnt'].shift(-1).fillna(method='ffill')

        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df