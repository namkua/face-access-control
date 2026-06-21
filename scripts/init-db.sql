-- Initialize databases for face-recognition-mlops
-- This script runs automatically when the postgres container starts for the first time.

-- Create the airflow database (Airflow requires a separate DB)
SELECT 'CREATE DATABASE airflow OWNER ' || current_user
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Grant all privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE airflow TO CURRENT_USER;
