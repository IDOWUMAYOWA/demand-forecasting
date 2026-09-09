from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "demand-forecasting",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="train_demand_forecasting_model",
    default_args=default_args,
    description="Retrain the CatBoost demand forecasting model",
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["demand-forecasting", "training"],
) as dag:

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