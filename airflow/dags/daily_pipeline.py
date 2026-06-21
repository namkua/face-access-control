"""
Airflow DAG for daily batch processing.

Orchestrates daily aggregation, data quality checks, and reporting.
"""
import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.exceptions import AirflowSkipException
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

# ── Infrastructure config (read from env, with docker-compose defaults) ──────
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC_ACCESS_LOGS = os.getenv('KAFKA_TOPIC_ACCESS_LOGS', 'access-logs')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET_RAW = os.getenv('MINIO_BUCKET_RAW', 'raw-images')
MINIO_BUCKET_PROCESSED = os.getenv('MINIO_BUCKET_PROCESSED', 'processed-data')

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Create DAG
dag = DAG(
    'daily_batch_processing',
    default_args=default_args,
    description='Daily batch processing for face recognition logs',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['batch', 'daily', 'mlops'],
)


# ═════════════════════════════════════════════════════════════════════════════
# Task callables
# ═════════════════════════════════════════════════════════════════════════════

def check_data_availability(**context):
    """
    Check if data is available for processing.

    List objects in the MinIO ``access-logs-lake`` bucket for the execution date
    to confirm new access logs were ingested.

    Raises ``AirflowSkipException`` when no new data is found,
    which gracefully skips all downstream tasks instead of failing the DAG.
    """
    from minio import Minio

    execution_date = context['execution_date']
    # Extract date components for partition path
    year = f"{execution_date.year:04d}"
    month = f"{execution_date.month:02d}"
    day = f"{execution_date.day:02d}"
    date_str = execution_date.strftime('%Y-%m-%d')
    logger.info(f"Checking data availability for {date_str} in MinIO Lake")

    minio_has_data = False
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
        
        # Check if the partition exists in the lake
        # access-logs-lake/access-logs/Year=YYYY/Month=MM/dd/
        prefix = f"access-logs/Year={year}/Month={month}/{day}/"
        objects = list(
            client.list_objects("access-logs-lake", prefix=prefix, recursive=True)
        )
        # We need at least one .parquet file
        json_files = [obj for obj in objects if obj.object_name.endswith('.json')]
        
        minio_has_data = len(json_files) > 0
        logger.info(
            f"MinIO bucket 'access-logs-lake' prefix '{prefix}': "
            f"{len(json_files)} json files found"
        )
    except Exception as exc:
        logger.warning(f"MinIO check failed (non-fatal): {exc}")

    # ── Decision ─────────────────────────────────────────────────────────
    if not minio_has_data:
        raise AirflowSkipException(
            f"No new parquet data found in data lake for {date_str} — skipping pipeline run"
        )

    context['ti'].xcom_push(key='data_status', value={
        'minio_has_data': minio_has_data,
        'date': date_str,
    })
    logger.info("Data availability check passed")
    return True



# ═════════════════════════════════════════════════════════════════════════════
# Task definitions
# ═════════════════════════════════════════════════════════════════════════════

# Task 1: Check data availability
check_data = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    provide_context=True,
    dag=dag,
)


# Task 2: Run Spark aggregation job
spark_aggregation = SparkSubmitOperator(
    task_id='spark_daily_aggregation',
    application='/opt/airflow/batch/spark_daily_aggregation.py',
    conn_id='spark_default',
    # Removed kafka package, kept delta and aws for S3/MinIO
    packages='io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.3.4',
    total_executor_cores=4,
    executor_cores=2,
    executor_memory='4g',
    driver_memory='2g',
    name='daily_aggregation',
    conf={
        'spark.sql.shuffle.partitions': '50',
        'spark.default.parallelism': '50'
    },
    env_vars={"SPARK_PROCESS_DATE": "{{ ds }}"},
    dag=dag,
)


# Task 3: Archive old data
#   Remove objects older than 90 days from the processed-data bucket using
#   the MinIO Client (mc).  The --older-than flag keeps the command simple
#   and avoids custom date arithmetic.
archive_data = BashOperator(
    task_id='archive_old_data',
    bash_command='''
        set -euo pipefail
        echo "=== Archiving data older than 90 days ==="

        # Configure MinIO client alias
        mc alias set myminio http://{{ params.minio_endpoint }} \
            {{ params.minio_access_key }} {{ params.minio_secret_key }} \
            --api S3v4 2>/dev/null || true

        BUCKET="myminio/{{ params.bucket_processed }}"

        # Count objects before cleanup
        BEFORE=$(mc find "${BUCKET}" --older-than 90d 2>/dev/null | wc -l || echo 0)
        echo "Objects older than 90 days: ${BEFORE}"

        if [ "${BEFORE}" -gt 0 ]; then
            mc rm --recursive --force --older-than 90d "${BUCKET}" 2>/dev/null \
                && echo "Archive cleanup completed" \
                || echo "WARNING: mc rm failed — manual cleanup may be needed"
        else
            echo "No objects to archive"
        fi

        echo "=== Archival complete ==="
    ''',
    params={
        'minio_endpoint': MINIO_ENDPOINT,
        'minio_access_key': MINIO_ACCESS_KEY,
        'minio_secret_key': MINIO_SECRET_KEY,
        'bucket_processed': MINIO_BUCKET_PROCESSED,
    },
    dag=dag,
)


# Define task dependencies
check_data  >> spark_aggregation >> archive_data 
