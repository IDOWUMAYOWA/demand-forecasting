import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

import pandas as pd
from typing import Dict, Any
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostprocessingPipeline


class PipelineRunner:
    """
    A class that orchestrates the execution of all stages in the ML pipeline.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline configurations.
            data_manager (DataManager): Instance for managing I/O operations on data.
        """
        self.config = config
        # Initialize individual pipeline components
        self.preprocessing_pipeline = PreprocessingPipeline(config=config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config=config)
        self.training_pipeline = TrainingPipeline(config=config)
        self.inference_pipeline = InferencePipeline(config=config)
        self.postprocessing_pipeline = PostprocessingPipeline(config=config)

    def run_training(self, df: pd.DataFrame) -> None:
        """
        Run the full training pipeline.

        df (pd.DataFrame): Input training DataFrame with features and target.

        Returns:
            None
        """
        df = self.preprocessing_pipeline.run(df=df)
        df = self.feature_eng_pipeline.run(df=df)
        model = self.training_pipeline.run(df)
        self.postprocessing_pipeline.run_train(model=model)
        return

    def run_inference(self, df: pd.DataFrame) -> None:
        """
        Run the full inference pipeline:
        Args:
            df (pd.DataFrame): The dataframe for which to run inference.

        Returns:
            None
        """
        # Step 1: Run preprocessing and feature engineering
        df = self.preprocessing_pipeline.run(df=df)
        df = self.feature_eng_pipeline.run(df=df)

        # Step 2: Run inference
        y_pred = self.inference_pipeline.run(x=df)

        # Step 3: Postprocessing and saving the prediction
        df_pred = self.postprocessing_pipeline.run_inference(
            y_pred=y_pred,
        )
        return df_pred