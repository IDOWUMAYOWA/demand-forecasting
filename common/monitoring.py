import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from scipy.stats import ks_2samp


class DriftMonitor:
    """
    Decides whether the production model should be retrained, based on
    (1) recent prediction performance degrading past a threshold, and
    (2) recent incoming feature distributions drifting from the data
    the model was originally trained on.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perf_cfg = config['monitoring']['performance']
        self.drift_cfg = config['monitoring']['data_drift']

    def log_prediction(self, predicted_at, target_hour, predicted_cnt) -> None:
        """
        Append a new prediction to the log so it can later be reconciled
        against the actual observed count once that hour has passed.
        """
        log_path = Path(self.perf_cfg['log_path'])
        entry = pd.DataFrame([{
            'predicted_at': predicted_at,
            'target_hour': target_hour,
            'predicted_cnt': predicted_cnt,
            'actual_cnt': np.nan,
        }])

        if log_path.exists():
            log = pd.concat([pd.read_parquet(log_path), entry], ignore_index=True)
        else:
            log = entry

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(log_path, index=False)

    def check_performance_drift(self) -> bool:
        """
        True if recent rolling MAE on reconciled predictions exceeds the
        configured threshold.
        """
        log_path = Path(self.perf_cfg['log_path'])
        if not log_path.exists():
            return False

        log = pd.read_parquet(log_path)
        reconciled = log.dropna(subset=['actual_cnt'])
        if len(reconciled) < self.perf_cfg['min_samples']:
            return False

        recent = reconciled.tail(self.perf_cfg['min_samples'])
        mae = np.mean(np.abs(recent['actual_cnt'] - recent['predicted_cnt']))
        return mae > self.perf_cfg['threshold']

    def check_data_drift(self, raw_data_path: str, prod_data_path: str) -> bool:
        """
        True if a Kolmogorov-Smirnov test finds any monitored feature's
        recent distribution significantly different from the training
        baseline's distribution.
        """
        prod_path = Path(prod_data_path)
        if not prod_path.exists():
            return False

        baseline = pd.read_parquet(raw_data_path)
        recent = pd.read_parquet(prod_path).tail(self.drift_cfg['window_hours'])
        if len(recent) < 24:
            return False

        for feature in self.drift_cfg['features']:
            if feature not in baseline.columns or feature not in recent.columns:
                continue
            _, p_value = ks_2samp(baseline[feature], recent[feature])
            if p_value < self.drift_cfg['p_value_threshold']:
                return True
        return False

    def should_retrain(self, raw_data_path: str, prod_data_path: str) -> bool:
        """
        Combined decision: retrain if EITHER signal fires.
        """
        return (
            self.check_performance_drift()
            or self.check_data_drift(raw_data_path, prod_data_path)
        )
    def reconcile_from_history(self, raw_data_path: str) -> None:
        """
        Fill in actual_cnt for logged predictions by matching target_hour
        against real historical data.
        """
        log_path = Path(self.perf_cfg['log_path'])
        if not log_path.exists():
            return

        log = pd.read_parquet(log_path)
        unreconciled = log[log['actual_cnt'].isna()]
        if unreconciled.empty:
            return

        history = pd.read_parquet(raw_data_path)
        if 'datetime' not in history.columns:
            return
        history['datetime'] = pd.to_datetime(history['datetime'])
        history = history.set_index('datetime')

        for idx, row in unreconciled.iterrows():
            target_hour = pd.Timestamp(row['target_hour']).floor('h')
            if target_hour in history.index:
                log.loc[idx, 'actual_cnt'] = history.loc[target_hour, 'cnt']

        log.to_parquet(log_path, index=False)