# Face Recognition MLOps System

A production-ready, enterprise-grade face recognition system featuring a complete MLOps pipeline and real-time processing running locally via Docker Compose. Built on modern open-source technologies including FastAPI, NVIDIA Triton Inference Server, Feast Feature Store, Kafka, Flink, Spark, Airflow, and Prometheus/Grafana, this project demonstrates a complete ML lifecycle from user onboarding and feature extraction through high-throughput serving and distributed observability.

<img src="images/pipeline.png" height="700" alt="Architecture">

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset & Schemas](#dataset--schemas)
  - [User Schema](#user-schema)
  - [Prediction Logs Schema](#prediction-logs-schema)
  - [Check-In Records Schema](#check-in-records-schema)
  - [Feature Engineering & Feature Store](#feature-engineering--feature-store)
- [Architecture Overview](#architecture-overview)
  - [1. Data & Streaming Pipeline](#1-data--streaming-pipeline)
    - [Data Sources](#data-sources)
    - [Input Validation](#input-validation)
    - [Storage Layer](#storage-layer)
    - [Stream Processing](#stream-processing)
  - [2. Orchestration & Batch Processing](#2-orchestration--batch-processing)
    - [Employee Onboarding](#employee-onboarding)
    - [Batch Aggregations](#batch-aggregations)
  - [3. Serving Pipeline](#3-serving-pipeline)
    - [Model Serving (Triton)](#model-serving-triton)
    - [Feature Service (Feast)](#feature-service-feast)
  - [4. Observability & Security](#4-observability--security)
    - [Metrics & Dashboards](#metrics--dashboards)
    - [Access & Security Management](#access--security-management)
- [Details](#details)
  - [Prerequisites & Model Export](#prerequisites--model-export)
  - [Setup Environment Variables](#setup-environment-variables)
  - [Start MLOps Services (Local)](#start-mlops-services-local)
  - [Initialize Database](#initialize-database)
  - [Interact with the Serving API](#interact-with-the-serving-api)
  - [Object Storage (MinIO)](#object-storage-minio)
  - [Real-Time Streaming & Fraud Detection (Flink)](#real-time-streaming--fraud-detection-flink)
  - [Start Orchestration & Batch Pipelines (Airflow & Spark)](#start-orchestration--batch-pipelines-airflow--spark)
  - [Start Observability](#start-observability)

---

## Project Structure

```
face-recognition-mlops/
 api/                        # FastAPI application
    main.py                # App entry point
    config.py              # Configuration
    models.py              # Database models
    schemas.py             # Pydantic schemas
    routes/                # API endpoints
    services/              # Business logic
    tests/                 # Unit tests
 airflow/                    # Airflow orchestration
    dags/                  # DAG definitions
       daily_pipeline.py  # Daily batch processing
       employee_onboarding_dag.py # Onboarding
    batch/                 # Apache Spark batch jobs
        spark_daily_aggregation.py
 features/                   # Feast feature store
    feast_features.py      # Feature definitions
    feast_client.py        # Client wrapper
    feature_store.yaml     # Feature store configs
 scripts/                    # Utility scripts
    data_generator.py      # Generate test data
    export_facenet_onnx.py # Convert FaceNet to ONNX
    feast_init.sh          # Feast DB initializer
    backup_restore.sh      # Backup utilities
 monitoring/                 # Monitoring configs
    prometheus/            # Prometheus setup
    grafana/               # Grafana dashboards
 docker-compose.yml          # Local development
 Jenkinsfile                 # CI/CD pipeline
 requirements.txt            # Python dependencies
 README.md                   # This file
```

---

## Dataset & Schemas

> Face Recognition Onboarding and Check-In Activity Records

Unlike standard tabular ML systems, the face recognition pipeline operates on face photos, 512-dimensional vector embeddings, and real-time transaction/access logs. The data is divided into registration info (PostgreSQL/Feast), detection and prediction history (PostgreSQL/MinIO), and transactional check-in events (Kafka/Delta Lake).

### User Schema

Stored in PostgreSQL for registered users/employees.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Unique auto-incrementing identifier |
| `uuid` | String | Unique UUID v4 for external references |
| `email` | String | Employee email address (unique index) |
| `username` | String | Unique username identifier (unique index) |
| `full_name` | String | Full name of the employee |
| `face_embedding` | JSON | 512-dimensional FaceNet vector embedding |
| `created_at` | DateTime | Timestamp when the user was registered |
| `updated_at` | DateTime | Timestamp when the user profile was last updated |

### Prediction Logs Schema

Captured on every prediction request to audit model inference performance.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Unique identifier |
| `uuid` | String | Unique request UUID |
| `user_id` | Integer (FK) | Reference to registered user (if recognized) |
| `image_path` | String | Path to raw uploaded image stored in MinIO |
| `confidence` | Float | Model output similarity confidence score |
| `processing_time_ms`| Float | Total processing latency in milliseconds |
| `status` | String | Outcome status: `success`, `no_face`, or `error` |
| `error_message` | String | Diagnostic error details if inference failed |
| `meta_data` | JSON | Extensible prediction metadata |
| `created_at` | DateTime | Timestamp when prediction was performed |

### Check-In Records Schema

Tracks employee attendance transactions, pushed to Kafka and synced to database/data lake.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Unique identifier |
| `uuid` | String | Unique event UUID |
| `user_id` | Integer (FK) | Reference to the recognized user |
| `location` | String | Location or gate identifier of check-in |
| `device_id` | String | Terminal device identifier |
| `confidence` | Float | Inference confidence score |
| `check_in_type` | String | Input medium: `face`, `manual`, or `card` |
| `meta_data` | JSON | Additional check-in context |
| `created_at` | DateTime | Attendance check-in timestamp |

### Feature Engineering & Feature Store

We utilize the **Feast Feature Store** to register, serve, and materialize employee facial embeddings. Real-time embeddings are pushed to the Redis Online Store for sub-5ms lookups during inference, while PostgreSQL stores the offline history for model validation.

Key registered features in Feast:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `embedding` | Array(Float32) | 512-dimensional FaceNet vector embedding |
| `embedding_model` | String | Model identification tag (e.g. `facenet`, `arcface`) |
| `embedding_version` | Int64 | Model version tag |
| `image_quality_score`| Float32 | Quality verification score of the registered photo |
| `num_faces_detected` | Int64 | Number of faces detected in registration image (must be 1) |
| `registration_confidence`| Float32 | Model confidence during registration |

---

## Architecture Overview


The face recognition system encompasses four main stages: **API & Serving**, **Data & Streaming**, **Orchestration & Batch Processing**, and **Observability & Security**.

### 1. Data & Streaming Pipeline

#### Data Sources
- **FastAPI Endpoint**: Receives uploaded check-in photos and writes raw images to MinIO storage.
- **Kafka Producer**: Streams check-in logs and security alerts (`tracking.access_logs`, `tracking.alerts`) for real-time downstream analytics.

#### Input Validation
- **OpenCV & MTCNN**: Performs image decoding, verifies face presence, and calculates image quality metrics before running model inference.
- **Pydantic**: Enforces strict payload validation on REST endpoints.

#### Storage Layer
- **MinIO Object Store**: Houses raw/processed images and serves as the Delta Lake storage layer.
- **PostgreSQL Database**: Holds user credentials, transaction prediction tables, and check-in logs.
- **Redis Online Store**: Manages real-time embeddings for instant cosine-similarity matching.

#### Stream Processing
- **Apache Flink**: Analyzes Kafka access logs in real-time, enforcing security policies (e.g., detecting brute-force login attempts and repeated check-in failures) and pushing alerts.

---

### 2. Orchestration & Batch Processing

#### Employee Onboarding
- **Apache Airflow**: Coordinates multi-stage on-demand employee registration workflows.
  - Validates employee data and uploads.
  - Extracts 512-dim facial vectors.
  - Pushes embeddings to Feast (Redis and PostgreSQL).
  - Verifies registration with test prediction tasks.

#### Batch Aggregations
- **Apache Spark**: Executes daily batch ETL pipelines over historical access logs, generating performance dashboards and writing structured results to Delta Lake tables.

---

### 3. Serving Pipeline

#### Model Serving (Triton)
- **NVIDIA Triton Inference Server**: Loads the exported FaceNet ONNX model. Uses dynamic batching (up to 32 requests) and CPU/GPU-friendly serving to achieve under 100ms inference latency.

#### Feature Service (Feast)
- **Feast Retrieval Client**: Retrieves registered embeddings from Redis online store with sub-5ms lookup latency.
- **Cosine Similarity Engine**: Compares Triton's output embedding with the Feast feature vector using numpy cosine-similarity to identify the employee (matching threshold: 0.6).

---

### 4. Observability & Security

#### Metrics & Dashboards
- **Prometheus & Grafana**: Scraping application metrics (latencies, counts, accuracy, confidence levels) and displaying pre-configured Grafana dashboards.
- **Jaeger**: Traces requests across all microservices (API, Triton, Redis, Postgres, MinIO).

#### Access & Security Management
- **Secrets**: Encrypted using `.env` configs.
- **Validation**: Pydantic input validation, SQLAlchemy ORM for SQL injection protection.

---

## Details

### Prerequisites & Model Export

Before starting the containers, NVIDIA Triton Inference Server requires the FaceNet model to be exported and structured correctly in the `./models/` directory.

1. Install the local python requirements (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the ONNX conversion script to fetch the pre-trained FaceNet weights and export them to ONNX format:
   ```bash
   python scripts/export_facenet_onnx.py
   ```
   *This creates a structured directory `models/facenet/1/model.onnx` along with the Triton configuration file `models/facenet/config.pbtxt`.*

---

### Setup Environment Variables

Configure your local environment parameters by copying the example file:
```bash
cp .env.example .env
```

Ensure the `.env` contents match your local ports and credentials:
```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
API_WORKERS=4

# Database & Cache
DATABASE_URL=postgresql://user:pass@localhost:5432/facedb
REDIS_URL=redis://localhost:6379/0

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Object Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Model Serving
MODEL_NAME=Facenet512
MODEL_THRESHOLD=0.6
TRITON_URL=localhost:8000

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_AGENT_HOST=localhost
```

---

### Start MLOps Services (Local)

Launch all core infrastructure, microservices, and metrics exporters using Docker Compose:
```bash
docker-compose up -d
```

Verify that all services are healthy and running:
```bash
docker-compose ps
```

---

### Initialize Database

Once the databases are active, initialize the schemas:

1. **Apply Alembic Migrations (Required):** Setup relational tables (`users`, `prediction_logs`, `check_in_records`) in PostgreSQL. This step is **manual** and must be run once:
   ```bash
   docker-compose exec api alembic upgrade head
   ```

---

### Interact with the Serving API

The FastAPI serving API provides endpoints to onboard employees, register face photos, and perform real-time verification.

![API Swagger UI](./images/APISwagger.png)

---

### Object Storage (MinIO)

MinIO acts as the local S3-compatible object storage layer for raw image storage, raw data lake telemetry, and the processed batch analytics catalog.

- **Access Console:** Open the MinIO web dashboard at `http://localhost:9001` (login credentials: `minioadmin` / `minioadmin`).
- **Buckets Configuration:**
  - `raw-images`: Stores all raw uploaded face photos from onboarding and check-in FastAPI calls.
  - `access-logs-lake`: Serves as the raw Data Lake storage. Kafka Connect S3 Sink streams access log telemetry events directly into this bucket (partitioned by Year/Month/Day) to be processed by Spark.
  - `processed-data`: Serves as the storage backend for Apache Spark batch ETL outputs and aggregated Delta Lake tables.

![MinIO Console](./images/MinIOConsole.png)

---

### Real-Time Streaming & Fraud Detection (Flink)

The Flink streaming engine monitors face check-in events in real-time to detect anomalous or fraudulent access behaviors.

1. **Streaming Logic:**
   - Every face prediction event publishes telemetry to Kafka on the `access-logs` topic.
   - The Apache Flink job (`streaming/fraud_detector/flink_fraud_detector.py`) consumes from this topic and tracks events using a sliding window.
   - If a user triggers repeated failed check-ins (e.g., 5 failures within 60 seconds), Flink flags it as a security event and publishes to the `alerts` topic.
2. **Accessing Web Console:**
   - Monitor job graph, slot allocation, and flamegraphs at the **Flink Web UI** (`http://localhost:8088`).

![Flink UI](./images/FlinkDashboard.png)

---

### Start Orchestration & Batch Pipelines (Airflow & Spark)

#### Apache Airflow (Workflow Orchestration)
Access the Airflow UI at `http://localhost:8081` (default credentials: `admin` / `admin`).
The system provisions the following DAGs:
- **`employee_onboarding`**: Orchestrates onboarding workflows (validates employee photos, calls Triton to extract embeddings, updates Feast online/offline stores, and generates validation logs).
- **`daily_batch_processing`**: Coordinates batch ETL processes, executing daily Spark aggregate analytics over check-in history logs.
- **`feast_materialize_incremental`**: Automatically materializes (syncs) features hourly from PostgreSQL (Offline Store) to Redis (Online Store) to ensure the fast online cache is kept up-to-date.

![Airflow UI](./images/AirflowDashboard.png)

#### Apache Spark (Batch Processing)
Spark is used to process long-term historical check-in logs and calculate aggregates (e.g., daily attendance reports, confidence decay analysis):
- Airflow schedules and triggers Spark jobs in the background.
- You can manually invoke the Spark daily aggregation script inside the environment:
  ```bash
  python airflow/batch/spark_daily_aggregation.py
  ```

---

### Start Observability

The platform provides comprehensive monitoring and tracing across the entire serving and processing lifecycle.

#### Observability Quick Links
| Service / Dashboard | Local URL | Description / Credentials |
| :--- | :--- | :--- |
| **Prometheus** | http://localhost:9090 | System and exporter metrics scraper |
| **Grafana** | http://localhost:3000 | Dashboards for API performance and model quality (`admin` / `admin`) |
| **Jaeger UI** | http://localhost:16686 | Distributed request tracing to isolate serving bottlenecks |


#### Metrics Scraping & Monitoring (Prometheus & Grafana)
- **Prometheus UI (`http://localhost:9090`)**: Scrapes performance metrics from FastAPI, Triton Inference Server, PostgreSQL Exporter, Redis Exporter, Kafka Exporter, and Flink metrics reporters.
- **Grafana Dashboards (`http://localhost:3000`)**:
  - Sign in using `admin` / `admin`.
  - The project comes with pre-configured dashboards displaying API throughput, serving latency percentiles, Triton server load, model inference confidence levels, and active Kafka message lags.
  - *Dashboards are auto-loaded from `monitoring/grafana/dashboards`.*

![Grafana Dashboard](./images/GrafanaDashboard.png)

#### Distributed Tracing (Jaeger)
- **Jaeger UI (`http://localhost:16686`)**:
  - Used for tracing end-to-end request latency.
  - To trace a request:
    1. Make a prediction call: `curl -X POST http://localhost:8000/api/v1/predict -F "file=@face_photo.jpg"`
    2. Open Jaeger UI, select service `face-recognition-api`, and click **Find Traces**.
    3. Click on the trace to view the exact breakdown of database queries, Triton model execution times, and Feast feature retrieval latencies.

![Jaeger UI](./images/JaegerUI.png)

#### Message Queue Monitoring (Kafka UI)
- **Kafka UI (`http://localhost:8080`)**:
  - Allows developers to browse and inspect Kafka topics.
  - Inspect messages in topics: `access-logs` (published by FastAPI for check-in attempts), `alerts` (published by Flink for anomalies), and `events` (published by FastAPI for admin events like user creation or registration triggers).
  - Track consumer group positions to verify if the Flink job or Kafka Connect tasks are lagging.

![Kafka UI](./images/KafkaUI.png)



