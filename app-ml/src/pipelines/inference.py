import pandas as pd
from typing import Dict, Any
from catboost import CatBoostRegressor


class InferencePipeline:
    """
    A pipeline for making predictions using a trained model.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.target_col = config['training']['target_col']
        self.model_path = config['data_manager']['model_path']
        self._model = None  # loaded lazily, on first use

    def _load_model(self) -> CatBoostRegressor:
        if self._model is None:
            model = CatBoostRegressor()
            model.load_model(self.model_path)
            self._model = model
        return self._model

    def run(self, x: pd.DataFrame) -> float:
        """
        Execute the complete inference pipeline.

        Args:
            x (pd.DataFrame): Input DataFrame covering a historical window plus
                the current row, so lag features are computed from real data.

        Returns:
            float: The prediction for the most recent row.
        """
        model = self._load_model()
        feature_cols = [c for c in x.columns if c != self.target_col]
        preds = model.predict(x[feature_cols])
        return float(preds[-1])