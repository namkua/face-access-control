"""
Script to generate large amounts of synthetic data for big data requirements.

Generates 500M+ check-in records to reach 100GB+ storage.
"""
import random
import json
from datetime import datetime, timedelta
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()


def generate_user_pool(num_users=10000):
    """Generate a pool of fake users."""
    logger.info(f"Generating {num_users} fake users...")
    
    users = []
    for i in range(num_users):
        user = {
            "user_id": str(uuid.uuid4()),
            "username": fake.user_name(),
            "email": fake.email(),
            "full_name": fake.name(),
            "department": random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']),
            "location": random.choice(['Office A', 'Office B', 'Office C', 'Remote'])
        }
        users.append(user)
    
    return users


def generate_check_in_batch(users, start_date, end_date, batch_size=1000000):
    """Generate a batch of check-in records conforming to the data lake schema."""
    
    records = []
    
    for _ in range(batch_size):
        user = random.choice(users)
        
        # Random timestamp between start and end date
        time_delta = end_date - start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        # Determine status
        status = random.choice(['success', 'success', 'success', 'no_match'])
        is_success = (status == 'success')
        matched_user = user["user_id"] if is_success else None
        user_id = user["user_id"] if is_success else "anonymous"
        confidence = float(random.uniform(0.6, 1.0)) if is_success else float(random.uniform(0.3, 0.6))
        
        # Generate metadata dict matching production app.py fields
        metadata = {
            "prediction_id": str(uuid.uuid4()),
            "matched_user": matched_user,
            "confidence": confidence,
            "processing_time_ms": float(random.uniform(50, 500)),
            "source_ip": fake.ipv4(),
            "prediction_status": status,
            # Additional attributes for completeness
            "device_id": f"device_{random.randint(1, 50)}",
            "check_in_type": random.choice(['face', 'card', 'manual'])
        }
        
        # New Spark-compatible schema: user_id, action, metadata (nested JSON object), timestamp
        record = {
            "user_id": user_id,
            "action": "prediction",
            "metadata": metadata,
            "timestamp": timestamp.isoformat()
        }
        
        records.append(record)
    
    return records


def save_to_json(records, output_path):
    """Save records to JSONLines file."""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def generate_large_dataset(
    output_dir="data/generated",
    total_records=500_000_000,
    batch_size=1_000_000,
    num_users=10_000
):
    """
    Generate large synthetic dataset.
    
    Args:
        output_dir: Output directory for Parquet files
        total_records: Total number of records to generate
        batch_size: Records per batch/file
        num_users: Number of unique users
    """
    logger.info("=" * 80)
    logger.info("LARGE DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Target total records: {total_records:,}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Number of unique users: {num_users:,}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate user pool
    users = generate_user_pool(num_users)
    
    # Date range: 2020-01-01 to 2025-12-31
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # Calculate number of batches
    num_batches = max(1, total_records // batch_size)
    
    logger.info(f"Will generate {num_batches} batch files")
    logger.info("Starting generation...")
    
    total_size_bytes = 0
    
    # Generate batches
    for batch_num in tqdm(range(num_batches), desc="Generating batches"):
        # Generate batch
        records = generate_check_in_batch(users, start_date, end_date, batch_size)
        
        # Output file path
        year = 2020 + (batch_num % 6)  # Distribute across years
        month = (batch_num % 12) + 1
        day = (batch_num % 28) + 1  # Distribute across days
        output_path = os.path.join(
            output_dir,
            f"Year={year}",
            f"Month={month:02d}",
            f"{day:02d}",
            f"batch_{batch_num:06d}.json"
        )
        
        # Create subdirectory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        save_to_json(records, output_path)
        
        # Track size
        file_size = os.path.getsize(output_path)
        total_size_bytes += file_size
        
        if (batch_num + 1) % 10 == 0:
            total_size_gb = total_size_bytes / (1024**3)
            logger.info(f"Progress: {batch_num + 1}/{num_batches} batches | "
                       f"Total size: {total_size_gb:.2f} GB")
    
    # Final statistics
    total_size_gb = total_size_bytes / (1024**3)
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total records generated: {total_records:,}")
    logger.info(f"Total files created: {num_batches}")
    logger.info(f"Total size: {total_size_gb:.2f} GB")
    logger.info(f"Output directory: {output_dir}")
    
    # Create metadata file
    metadata = {
        "total_records": total_records,
        "total_files": num_batches,
        "total_size_gb": total_size_gb,
        "num_users": num_users,
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "generated_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    # Default to 1M records for local testing. Use TOTAL_RECORDS env var to override (e.g. 500M)
    total_records_env = int(os.getenv("TOTAL_RECORDS", "1000000"))
    batch_size_env = min(total_records_env, 100000)
    num_users_env = min(total_records_env // 10, 10000)
    if num_users_env < 10:
        num_users_env = 10
        
    generate_large_dataset(
        output_dir="data/generated",
        total_records=total_records_env,
        batch_size=batch_size_env,
        num_users=num_users_env
    )
