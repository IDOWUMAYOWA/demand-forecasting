import pandas as pd
from typing import Dict, Any
from catboost import CatBoostRegressor


class TrainingPipeline:
    """
    A pipeline class for training and optimizing machine learning model
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        train_cfg = config['training']
        self.target_col = train_cfg['target_col']
        self.model_params = train_cfg['model_params']

    def run(self, df: pd.DataFrame) -> Any:
        """
        Run the full training pipeline

        Args:
            df (pd.DataFrame): Input training DataFrame with features and target.

        Returns:
            Any: Trained model
        """
        feature_cols = [c for c in df.columns if c != self.target_col]
        x_train = df[feature_cols]
        y_train = df[self.target_col]

        model = CatBoostRegressor(**self.model_params)
        model.fit(x_train, y_train)

        return model