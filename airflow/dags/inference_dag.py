from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator

default_args = {
    "owner": "demand-forecasting",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="run_demand_forecasting_inference",
    default_args=default_args,
    description="Call the inference API hourly, logging each prediction for drift monitoring",
    schedule_interval="@hourly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["demand-forecasting", "inference"],
) as dag:

    run_inference = SimpleHttpOperator(
        task_id="run_inference",
        http_conn_id="inference_api",
        endpoint="/run-inference",
        method="POST",
        log_response=True,
    )