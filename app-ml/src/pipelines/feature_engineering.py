import pandas as pd
from typing import Dict, Any


class FeatureEngineeringPipeline:
    """
    A pipeline for creating and engineering features from preprocessed data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        fe_cfg = config['feature_engineering']
        self.target_lag_col = fe_cfg['target_lag_col']
        self.target_lags = fe_cfg['target_lags']
        self.lag_features = fe_cfg['lag_features']
        self.n_feature_lags = fe_cfg['n_feature_lags']

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be processed

        Returns:
            pd.DataFrame: DataFrame with engineered features including lag features
        """
        df = df.copy()

        # Target/cnt lags (captures daily seasonality: lag 1-2 = recent hours,
        # lag 22-23 = same time yesterday)
        for lag in self.target_lags:
            df[f'{self.target_lag_col}_lag_{lag}'] = (
                df[self.target_lag_col].shift(lag).bfill()
            )

        # Lags for other predictive features
        for feat in self.lag_features:
            for lag in range(1, self.n_feature_lags + 1):
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag).bfill()

        return df