from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd


class PostprocessingPipeline:
    """
    Handles postprocessing steps in the machine learning pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config['data_manager']['model_path']

    def run_train(self, model: Any) -> None:
        """
        Save the trained model to the file path specified in the config.
        """
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self.model_path)
        return

    def run_inference(self, y_pred: float) -> pd.DataFrame:
        """
        Format the model prediction as a single-row DataFrame for saving or
        further processing.

        Args:
            y_pred (float): The predicted value from the inference step.

        Returns:
            pd.DataFrame: Single-row DataFrame with the prediction and
                the timestamps it was generated for/at.
        """

        predicted_cnt = max(0, round(y_pred))
        now = datetime.utcnow()
        df_postprocessed = pd.DataFrame({
            'predicted_at': [now],
            'target_hour': [now + timedelta(hours=1)],
            'predicted_cnt': [predicted_cnt],
        })
        return df_postprocessed