# 🚀 Face Recognition MLOps System

A production-ready, enterprise-grade face recognition system featuring a complete MLOps pipeline and real-time processing running locally via Docker Compose. Built on modern open-source technologies including FastAPI, NVIDIA Triton Inference Server, Feast Feature Store, Kafka, Flink, Spark, Airflow, and Prometheus/Grafana, this project demonstrates a complete ML lifecycle from user onboarding and feature extraction through high-throughput serving and distributed observability.

![Architecture](image - Place the System Architecture Diagram here (e.g. ./docs/pipeline.png or docs/architecture.png))

## 📑 Table of Contents

- [📊 Dataset & Schemas](#-dataset--schemas)
  - [User Schema](#user-schema)
  - [Prediction Logs Schema](#prediction-logs-schema)
  - [Check-In Records Schema](#check-in-records-schema)
  - [Feature Engineering & Feature Store](#feature-engineering--feature-store)
- [🌐 Architecture Overview](#-architecture-overview)
  - [1. Data & Streaming Pipeline](#1-data--streaming-pipeline)
    - [📤 Data Sources](#-data-sources)
    - [✅ Input Validation](#-input-validation)
    - [☁️ Storage Layer](#-storage-layer)
    - [🛒 Stream Processing](#-stream-processing)
  - [2. Orchestration & Batch Processing](#2-orchestration--batch-processing)
    - [🌟 Employee Onboarding](#-employee-onboarding)
    - [📦 Batch Aggregations](#-batch-aggregations)
  - [3. Serving Pipeline](#3-serving-pipeline)
    - [⚡ Model Serving (Triton)](#-model-serving-triton)
    - [🔍 Feature Service (Feast)](#-feature-service-feast)
  - [4. Observability & Security](#4-observability--security)
    - [📡 Metrics & Dashboards](#-metrics--dashboards)
    - [🔒 Access & Security Management](#-access--security-management)
- [📖 Details](#-details)
  - [🔧 Setup Environment Variables](#-setup-environment-variables)
  - [🏁 Start MLOps Services (Local)](#-start-mlops-services-local)
  - [✅ Initialize Database & Feature Store](#-initialize-database--feature-store)
  - [🚀 Interact with the Serving Pipeline](#-interact-with-the-serving-pipeline)
  - [🔄 Start Orchestration & Batch Pipelines](#-start-orchestration--batch-pipelines)
  - [🔎 Start Observability](#-start-observability)
  - [📁 Project Structure](#-project-structure)
  - [🧪 Testing](#-testing)
  - [💡 Tips & Troubleshooting](#-tips--troubleshooting)
- [🤝 Contributing](#-contributing)
- [📃 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)

---

## 📊 Dataset & Schemas

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

## 🌐 Architecture Overview

The face recognition system encompasses four main stages: **API & Serving**, **Data & Streaming**, **Orchestration & Batch Processing**, and **Observability & Security**.

![Architecture Details](image - Suggestion: Place a detailed architectural workflow or dataflow diagram here (e.g., ./docs/detailed_architecture.png))

### 1. Data & Streaming Pipeline

#### 📤 Data Sources
- **FastAPI Endpoint**: Receives uploaded check-in photos and writes raw images to MinIO storage.
- **Kafka Producer**: Streams check-in logs and security alerts (`tracking.access_logs`, `tracking.alerts`) for real-time downstream analytics.

#### ✅ Input Validation
- **OpenCV & MTCNN**: Performs image decoding, verifies face presence, and calculates image quality metrics before running model inference.
- **Pydantic**: Enforces strict payload validation on REST endpoints.

#### ☁️ Storage Layer
- **MinIO Object Store**: Houses raw/processed images and serves as the Delta Lake storage layer.
- **PostgreSQL Database**: Holds user credentials, transaction prediction tables, and check-in logs.
- **Redis Online Store**: Manages real-time embeddings for instant cosine-similarity matching.

#### 🛒 Stream Processing
- **Apache Flink**: Analyzes Kafka access logs in real-time, enforcing security policies (e.g., detecting brute-force login attempts and repeated check-in failures) and pushing alerts.

---

### 2. Orchestration & Batch Processing

#### 🌟 Employee Onboarding
- **Apache Airflow**: Coordinates multi-stage on-demand employee registration workflows.
  - Validates employee data and uploads.
  - Extracts 512-dim facial vectors.
  - Pushes embeddings to Feast (Redis and PostgreSQL).
  - Verifies registration with test prediction tasks.

#### 📦 Batch Aggregations
- **Apache Spark**: Executes daily batch ETL pipelines over historical access logs, generating performance dashboards and writing structured results to Delta Lake tables.

---

### 3. Serving Pipeline

#### ⚡ Model Serving (Triton)
- **NVIDIA Triton Inference Server**: Loads the exported FaceNet ONNX model. Uses dynamic batching (up to 32 requests) and CPU/GPU-friendly serving to achieve under 100ms inference latency.

#### 🔍 Feature Service (Feast)
- **Feast Retrieval Client**: Retrieves registered embeddings from Redis online store with sub-5ms lookup latency.
- **Cosine Similarity Engine**: Compares Triton's output embedding with the Feast feature vector using numpy cosine-similarity to identify the employee (matching threshold: 0.6).

---

### 4. Observability & Security

#### 📡 Metrics & Dashboards
- **Prometheus & Grafana**: Scraping application metrics (latencies, counts, accuracy, confidence levels) and displaying pre-configured Grafana dashboards.
- **Jaeger**: Traces requests across all microservices (API, Triton, Redis, Postgres, MinIO).

#### 🔒 Access & Security Management
- **Secrets**: Encrypted using `.env` configs.
- **Validation**: Pydantic input validation, SQLAlchemy ORM for SQL injection protection.

---

## 📖 Details

### 🔧 Setup Environment Variables

Create a `.env` file in the root directory:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/facedb
REDIS_URL=redis://localhost:6379/0

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Object Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Model
MODEL_NAME=Facenet512
MODEL_THRESHOLD=0.6
TRITON_URL=localhost:8000

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_AGENT_HOST=localhost
```

### 🏁 Start MLOps Services (Local)

Launch directly with Docker Compose:

```bash
docker-compose up -d
docker-compose ps
```

### ✅ Initialize Database & Feature Store

After starting your services, apply Alembic migrations to setup the PostgreSQL tables, and apply the Feast configurations to register feature views.

```bash
# 1. Run database migrations
docker-compose exec api alembic upgrade head

# 2. Initialize and apply Feast schema
docker-compose exec api /bin/bash -c "cd /app/features && feast apply"
```

### 🚀 Interact with the Serving Pipeline

Perform your first API requests to verify endpoints:

```bash
# 1. Register an employee user account
curl -X POST http://localhost:8000/api/v1/users \
  -F "email=admin@example.com" \
  -F "username=admin" \
  -F "full_name=Admin User"

# 2. Register a face photo for the employee
curl -X POST http://localhost:8000/api/v1/users/admin/register-face \
  -F "file=@face_photo.jpg"

# 3. Request face prediction check-in
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@face_photo.jpg"
```

### 🔄 Start Orchestration & Batch Pipelines

#### Apache Airflow (Workflow Orchestration)
Access the Airflow UI at `http://localhost:8081` (admin/admin).
The DAGs are defined under `airflow/dags/`:
- `employee_onboarding_dag`: Automates the registration pipelines.
- `daily_pipeline`: Runs batch analytical ETL jobs.

#### Apache Spark (Batch Processing)
Spark jobs run via Airflow or can be invoked locally for batch analytics:
```bash
# Trigger local Spark daily aggregate report
python airflow/batch/spark_daily_aggregation.py
```

### 🔎 Start Observability

#### Prometheus & Grafana
Verify that metrics scraping is operational and access dashboards.

| Observability Endpoint | URL | Credentials |
| :--- | :--- | :--- |
| **Prometheus Metrics** | http://localhost:9090 | - |
| **Grafana Dashboard** | http://localhost:3000 | `admin` / `admin` |
| **Kafka UI** | http://localhost:8080 | - |
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin` |
| **Flink Web UI** | http://localhost:8088 | - |

![Grafana Dashboards](image - Suggestion: Place a screenshot of Grafana dashboards showing API metrics, prediction confidence, and system health here (e.g. ./docs/grafana_dashboards.png))

#### Jaeger (Distributed Tracing)
Enables profiling of end-to-end inference workflows to isolate latency bottlenecks. Access the Jaeger query portal at `http://localhost:16686`.

---

### 📁 Project Structure

```
face-recognition-mlops/
├── api/                        # FastAPI application
│   ├── main.py                # App entry point
│   ├── config.py              # Configuration
│   ├── models.py              # Database models
│   ├── schemas.py             # Pydantic schemas
│   ├── routes/                # API endpoints
│   ├── services/              # Business logic
│   └── tests/                 # Unit tests
├── airflow/                    # Airflow orchestration
│   ├── dags/                  # DAG definitions
│   │   ├── daily_pipeline.py  # Daily batch processing
│   │   └── employee_onboarding_dag.py # Onboarding
│   └── batch/                 # Apache Spark batch jobs
│       └── spark_daily_aggregation.py
├── features/                   # Feast feature store
│   ├── feast_features.py      # Feature definitions
│   ├── feast_client.py        # Client wrapper
│   └── feature_store.yaml     # Feature store configs
├── scripts/                    # Utility scripts
│   ├── data_generator.py      # Generate test data
│   ├── export_facenet_onnx.py # Convert FaceNet to ONNX
│   ├── feast_init.sh          # Feast DB initializer
│   └── backup_restore.sh      # Backup utilities
├── monitoring/                 # Monitoring configs
│   ├── prometheus/            # Prometheus setup
│   └── grafana/               # Grafana dashboards
├── docker-compose.yml          # Local development
├── Jenkinsfile                 # CI/CD pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### 🧪 Testing

Run code quality, unit, integration, and load tests:

```bash
# Run pytest unit tests
pytest

# Generate coverage reports
pytest --cov=api --cov-report=html

# Run specific unit test files
pytest api/tests/unit/test_face_recognition.py

# Run integration tests
pytest api/tests/integration/

# Execute load testing using Locust
locust -f api/tests/load_tests.py
```

### 💡 Tips & Troubleshooting

#### Quick Diagnostic Commands
```bash
# View live API service logs
docker-compose logs -f api

# Verify Triton Inference Server status
curl http://localhost:8000/v2/health/ready

# Check registered Feast feature views
feast feature-views list
```

#### Common Issues

**Issue: API container crashes during startup**
Verify database connections and migrations:
```bash
docker-compose logs api
docker-compose exec api python -c "from api.database import engine; print(engine)"
```

**Issue: Low prediction/recognition accuracy**
Adjust confidence matching threshold parameters in your `.env` file:
```bash
# Decrease threshold (e.g. 0.5) to allow more matches
MODEL_THRESHOLD=0.5
```

**Issue: High latency during peaks**
Enable Triton dynamic batching:
```yaml
# Add dynamic batching configuration in models/facenet/config.pbtxt
dynamic_batching { }
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📃 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Face recognition library
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) - High-performance model serving
- [Feast Feature Store](https://github.com/feast-dev/feast) - MLOps features engine
- [FastAPI](https://github.com/tiangolo/fastapi) - Modern web API framework

---

## 📞 Support

- 📧 Email: support@example.com
- 💬 Slack: `#face-recognition`
- 📝 Issues: [GitHub Issues](https://github.com/your-org/face-recognition-mlops/issues)

**Built with ❤️ for production-grade MLOps**
