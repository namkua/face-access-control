from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# This DAG runs every hour to ensure the Redis Online Store is always 
# synchronized with the Postgres Offline Store (acting as a fallback/consistency check)
with DAG(
    'feast_materialize_incremental',
    default_args=default_args,
    description='A scheduled DAG to materialize features into Feast Redis Online Store',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    catchup=False,
    tags=['feast', 'feature_store', 'mlops'],
) as dag:

    # Install Feast and run the materialize-incremental command directly
    # This avoids Docker-in-Docker (DooD) socket permission issues.
    materialize_task = BashOperator(
        task_id='materialize_features',
        bash_command='''
            export PATH=$PATH:~/.local/bin
            pip install --no-cache-dir "feast[postgres,redis]==0.34.1" structlog && \
            cd /feast_repo && \
            feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
        '''
    )

    materialize_task
