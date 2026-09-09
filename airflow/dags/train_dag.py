from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "demand-forecasting",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift(**context):
    import os
    import sys
    os.chdir('/opt/airflow/project')
    sys.path.append('/opt/airflow/project')
    from common.utils import read_config
    from common.monitoring import DriftMonitor

    config = read_config('/opt/airflow/project/config/config.yaml')
    monitor = DriftMonitor(config=config)

    monitor.reconcile_from_history(config['data_manager']['raw_data_path'])
    should_retrain = monitor.should_retrain(
        raw_data_path=config['data_manager']['raw_data_path'],
        prod_data_path=config['data_manager']['prod_data_path'],
    )
    print(f"should_retrain = {should_retrain}")
    return should_retrain

with DAG(
    dag_id="train_demand_forecasting_model",
    default_args=default_args,
    description="Conditionally retrain the CatBoost demand forecasting model",
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["demand-forecasting", "training"],
) as dag:

    check_drift_task = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    train_model = DockerOperator(
        task_id="train_model",
        image="demand-forecasting-ml",
        api_version="auto",
        auto_remove=True,
        command="python app-ml/entrypoint/train.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source="/home/ims24/Demand_forecasting/data", target="/app/data", type="bind"),
            Mount(source="/home/ims24/Demand_forecasting/models", target="/app/models", type="bind"),
        ],
    )

    check_drift_task >> train_model