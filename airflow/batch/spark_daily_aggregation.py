"""
Apache Spark batch job for daily aggregation.

Processes access logs and generates daily reports.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, min, max, window, to_date, 
    from_json, current_timestamp, lit
)
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Create Spark session with necessary configurations."""
    return SparkSession.builder \
        .appName("Face Recognition Daily Aggregation") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
        .getOrCreate()


def process_access_logs(spark, date_str=None):
    """
    Process access logs for a specific date.
    
    Args:
        spark: SparkSession
        date_str: Date string in YYYY-MM-DD format (defaults to yesterday)
    """
    if date_str is None:
        # Check environment variable first, default to yesterday
        import os
        date_str = os.getenv("SPARK_PROCESS_DATE")
        if not date_str:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"Processing access logs for date: {date_str}")
    
    # Extract date components for partition path
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year = f"{dt.year:04d}"
    month = f"{dt.month:02d}"
    day = f"{dt.day:02d}"
    
    # Read Parquet/JSON from MinIO Data Lake
    lake_path = f"s3a://access-logs-lake/access-logs/Year={year}/Month={month}/{day}/"
    logger.info(f"Reading logs from Data Lake: {lake_path}")
    
    try:
        logs_df = spark.read.json(lake_path)
    except Exception as e:
        logger.warning(f"No data found in {lake_path} or error reading: {e}")
        # Create an empty dataframe with the expected schema if no data
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("metadata", StringType(), True),
            StructField("timestamp", StringType(), True)
        ])
        logs_df = spark.createDataFrame([], schema)
    
    logger.info(f"Total records for {date_str}: {logs_df.count()}")
    
    # Aggregation: Prediction metrics
    prediction_logs = logs_df.filter(col("action") == "prediction")
    
    # Save results directly to reports folder
    output_base = "s3a://processed-data/reports/"
    
    # Generate summary report
    summary = {
        "date": date_str,
        "total_events": logs_df.count(),
        "unique_users": logs_df.select("user_id").distinct().count(),
        "total_predictions": prediction_logs.count(),
        "processed_at": datetime.now().isoformat()
    }
    
    logger.info(f"Daily summary: {summary}")
    
    # Save summary as Parquet (force single file with coalesce(1))
    summary_df = spark.createDataFrame([summary])
    summary_path = f"{output_base}/date={date_str}"
    summary_df.coalesce(1).write \
        .mode("overwrite") \
        .format("parquet") \
        .save(summary_path)
    
    return summary


def main():
    """Main entry point for Spark job."""
    logger.info("Starting Spark Daily Aggregation Job...")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Process logs for yesterday
        summary = process_access_logs(spark)
        
        logger.info("Job completed successfully!")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
