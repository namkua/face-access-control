"""
Airflow DAG for employee onboarding.

Replaces the Kubeflow employee onboarding pipeline. Orchestrates the process
of registering a new employee's face in the recognition system.
"""
import os
import io
import json
import tempfile
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Infrastructure config ────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET_RAW = os.getenv('MINIO_BUCKET_RAW', 'raw-images')

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'face_recognition')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'admin')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'changeme123')

API_BASE_URL = os.getenv('API_BASE_URL', 'http://api:8000')
FEAST_SERVER_URL = os.getenv('FEAST_SERVER_URL', 'http://feast:6566')

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

# ── Face quality thresholds ──────────────────────────────────────────────────
MIN_FACE_CONFIDENCE = 0.90
MIN_IMAGE_SIZE = (80, 80)          # Minimum face bounding-box pixels
MAX_INTER_EMBEDDING_DISTANCE = 0.4  # Max cosine distance between embeddings

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# Create DAG (triggered externally, no schedule)
dag = DAG(
    'employee_onboarding',
    default_args=default_args,
    description='Register employee face: validate images, extract embeddings, store in DB',
    schedule_interval=None,  # Triggered externally via API or Airflow CLI
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=3,  # Allow multiple onboarding processes in parallel
    tags=['onboarding', 'employee', 'mlops'],
)


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def _get_minio_client():
    """Return a configured Minio client."""
    from minio import Minio
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def _get_pg_connection():
    """Return a psycopg2 connection to the application database."""
    import psycopg2
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=int(POSTGRES_PORT),
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Task callables
# ═════════════════════════════════════════════════════════════════════════════

def validate_employee_data(**context):
    """
    Validate employee information and images.

    Expects DAG run conf:
    {
        "username": "john_doe",
        "email": "john@example.com",
        "full_name": "John Doe",
        "image_paths": ["s3://raw-images/onboarding/john_001.jpg", ...]
    }
    """
    conf = context['dag_run'].conf or {}

    user_id = conf.get('user_id')
    username = conf.get('username')
    email = conf.get('email')
    full_name = conf.get('full_name')
    image_paths = conf.get('image_paths', [])

    if not all([user_id, username, email, full_name]):
        raise ValueError("Missing required fields: user_id, username, email, full_name")

    if len(image_paths) < 1:
        raise ValueError(f"Need at least 1 image, got {len(image_paths)}")

    logger.info(f"Validating onboarding for {username} with {len(image_paths)} images")

    context['ti'].xcom_push(key='employee_data', value={
        "user_id": user_id,
        "username": username,
        "email": email,
        "full_name": full_name,
        "image_paths": image_paths,
    })

    return True


def detect_and_validate_faces(**context):
    """
    Run face detection on all provided images.

    For each image:
      1. Download from MinIO
      2. Run face detection (MTCNN from facenet-pytorch)
      3. Validate exactly one face is detected per image
      4. Check face quality (minimum size, confidence)
    """
    from facenet_pytorch import MTCNN
    from PIL import Image
    import torch

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    ti = context['ti']
    employee_data = ti.xcom_pull(task_ids='validate_employee_data', key='employee_data')
    image_paths = employee_data['image_paths']

    logger.info(f"Detecting faces in {len(image_paths)} images")

    minio_client = _get_minio_client()
    valid_images = []
    rejection_reasons = []

    for path in image_paths:
        # ── 1. Download from MinIO ───────────────────────────────────────
        # image_paths are like "s3://raw-images/onboarding/john_001.jpg"
        # Strip the s3://<bucket>/ prefix to get the object name
        if path.startswith('s3://'):
            parts = path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            object_name = parts[1] if len(parts) > 1 else ''
        else:
            bucket = MINIO_BUCKET_RAW
            object_name = path

        try:
            response = minio_client.get_object(bucket, object_name)
            image_bytes = response.read()
            response.close()
            response.release_conn()
        except Exception as exc:
            reason = f"{path}: download failed — {exc}"
            logger.warning(reason)
            rejection_reasons.append(reason)
            continue

        # ── 2. Run face detection ────────────────────────────────────────
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            boxes, probs = mtcnn.detect(pil_image)

            if boxes is None:
                faces = []
            else:
                faces = [
                    {
                        'confidence': float(probs[i]),
                        'facial_area': {
                            'x': int(boxes[i][0]),
                            'y': int(boxes[i][1]),
                            'w': int(boxes[i][2] - boxes[i][0]),
                            'h': int(boxes[i][3] - boxes[i][1]),
                        }
                    }
                    for i in range(len(boxes))
                ]
        except Exception as exc:
            reason = f"{path}: face detection error — {exc}"
            logger.warning(reason)
            rejection_reasons.append(reason)
            continue

        # ── 3. Validate single face ──────────────────────────────────────
        if not faces or len(faces) == 0:
            reason = f"{path}: no face detected"
            logger.warning(reason)
            rejection_reasons.append(reason)
            continue

        if len(faces) > 1:
            # Filter out low-confidence detections before rejecting
            high_conf = [f for f in faces if f.get('confidence', 0) >= MIN_FACE_CONFIDENCE]
            if len(high_conf) != 1:
                reason = f"{path}: expected 1 face, found {len(high_conf)} (high-confidence)"
                logger.warning(reason)
                rejection_reasons.append(reason)
                continue
            face = high_conf[0]
        else:
            face = faces[0]

        # ── 4. Check face quality ────────────────────────────────────────
        confidence = face.get('confidence', 0)
        if confidence < MIN_FACE_CONFIDENCE:
            reason = f"{path}: face confidence too low ({confidence:.3f} < {MIN_FACE_CONFIDENCE})"
            logger.warning(reason)
            rejection_reasons.append(reason)
            continue

        fa = face.get('facial_area', {})
        fw, fh = fa.get('w', 0), fa.get('h', 0)
        if fw < MIN_IMAGE_SIZE[0] or fh < MIN_IMAGE_SIZE[1]:
            reason = f"{path}: face too small ({fw}x{fh} < {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]})"
            logger.warning(reason)
            rejection_reasons.append(reason)
            continue

        valid_images.append(path)
        logger.info(f"{path}: passed (confidence={confidence:.3f}, size={fw}x{fh})")

    if len(valid_images) < 1:
        raise ValueError(
            f"Only {len(valid_images)} images passed face detection (need >= 1). "
            f"Rejected: {rejection_reasons}"
        )

    context['ti'].xcom_push(key='valid_images', value=valid_images)
    logger.info(f"{len(valid_images)}/{len(image_paths)} images passed validation")

    return True


def extract_embeddings(**context):
    """
    Extract face embeddings from validated images.

    Uses facenet-pytorch (InceptionResnetV1 pretrained on vggface2) to extract
    512-d embeddings and computes a representative embedding (median) for the
    employee.  Also validates that all embeddings are close to each other
    (same person).
    """
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    import torch

    device = torch.device('cpu')
    mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=True)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    ti = context['ti']
    employee_data = ti.xcom_pull(task_ids='validate_employee_data', key='employee_data')
    valid_images = ti.xcom_pull(task_ids='detect_validate_faces', key='valid_images')

    logger.info(f"Extracting embeddings from {len(valid_images)} images")

    minio_client = _get_minio_client()
    embeddings = []

    for path in valid_images:
        # Download image
        if path.startswith('s3://'):
            parts = path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            object_name = parts[1] if len(parts) > 1 else ''
        else:
            bucket = MINIO_BUCKET_RAW
            object_name = path

        response = minio_client.get_object(bucket, object_name)
        image_bytes = response.read()
        response.close()
        response.release_conn()

        # ── 1. Preprocess (detect + align via MTCNN) ─────────────────────
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # ── 2. Extract embedding ─────────────────────────────────────────
        face_tensor = mtcnn(pil_image)

        if face_tensor is None:
            logger.warning(f"No face detected for embedding in {path}")
            continue

        face_batch = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb_tensor = facenet(face_batch)

        emb = emb_tensor.squeeze(0).cpu().numpy()
        embeddings.append(emb)
        logger.info(f"Embedding extracted for {path}, dim={len(emb)}")

    if len(embeddings) < 1:
        raise ValueError(f"Only {len(embeddings)} embeddings extracted (need >= 1)")

    # ── 3. Validate same person (pairwise cosine similarity) ─────────────
    if len(embeddings) > 1:
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                ))
                cos_dist = 1.0 - cos_sim
                if cos_dist > MAX_INTER_EMBEDDING_DISTANCE:
                    raise ValueError(
                        f"Images {i} and {j} may not be the same person "
                        f"(cosine_distance={cos_dist:.4f} > {MAX_INTER_EMBEDDING_DISTANCE})"
                    )

    # ── 4. Pool embeddings (element-wise median) ─────────────────────────
    if len(embeddings) > 1:
        stacked = np.stack(embeddings)
        representative_embedding = np.median(stacked, axis=0).tolist()
    else:
        representative_embedding = embeddings[0].tolist()

    context['ti'].xcom_push(key='embedding', value=representative_embedding)
    logger.info(f"Representative embedding computed, dimension: {len(representative_embedding)}")

    return True


def register_in_database(**context):
    """
    Register the employee and their embedding in the database.

    Creates the user record (or updates if already exists) and stores
    the face embedding in the ``users`` table via direct SQL.
    """
    ti = context['ti']
    employee_data = ti.xcom_pull(task_ids='validate_employee_data', key='employee_data')
    embedding = ti.xcom_pull(task_ids='extract_embeddings', key='embedding')

    username = employee_data['username']
    email = employee_data['email']
    full_name = employee_data['full_name']

    logger.info(f"Registering {username} in database")

    conn = _get_pg_connection()
    try:
        cur = conn.cursor()

        # Check if user already exists
        cur.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        existing = cur.fetchone()

        embedding_json = json.dumps(embedding)

        if existing:
            # Update existing user's embedding
            cur.execute(
                """
                UPDATE users
                SET face_embedding = %s::jsonb,
                    full_name = %s,
                    updated_at = NOW()
                WHERE username = %s OR email = %s
                """,
                (embedding_json, full_name, username, email),
            )
            logger.info(f"Updated existing user {username} (id={existing[0]})")
        else:
            # Create new user
            cur.execute(
                """
                INSERT INTO users (uuid, email, username, full_name, face_embedding, created_at, updated_at)
                VALUES (gen_random_uuid()::text, %s, %s, %s, %s::jsonb, NOW(), NOW())
                RETURNING id
                """,
                (email, username, full_name, embedding_json),
            )
            new_id = cur.fetchone()[0]
            logger.info(f"Created new user {username} (id={new_id})")

        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Database registration failed: {exc}")
        raise
    finally:
        conn.close()

    logger.info(f"Employee {username} registered successfully")
    return True


def verify_registration(**context):
    """
    Verify that the new employee can be recognized.

    Runs a test prediction using one of the onboarding images by sending it
    to the ``/api/v1/predict`` endpoint and verifying the returned
    ``predicted_user`` matches the registered username.
    """
    import requests as req

    ti = context['ti']
    employee_data = ti.xcom_pull(task_ids='validate_employee_data', key='employee_data')
    valid_images = ti.xcom_pull(task_ids='detect_validate_faces', key='valid_images')
    username = employee_data['username']
    user_id = str(employee_data.get('user_id'))

    logger.info(f"Verifying registration for {username} (ID: {user_id})")

    # Download the first valid image for verification
    test_image_path = valid_images[0]
    minio_client = _get_minio_client()

    if test_image_path.startswith('s3://'):
        parts = test_image_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        object_name = parts[1] if len(parts) > 1 else ''
    else:
        bucket = MINIO_BUCKET_RAW
        object_name = test_image_path

    response = minio_client.get_object(bucket, object_name)
    image_bytes = response.read()
    response.close()
    response.release_conn()

    # Send test prediction to API
    predict_url = f"{API_BASE_URL}/api/v1/predict"
    try:
        resp = req.post(
            predict_url,
            files={'file': ('test_verify.jpg', io.BytesIO(image_bytes), 'image/jpeg')},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        predicted_user = result.get('predicted_user')
        confidence = result.get('confidence')
        status = result.get('status')

        logger.info(
            f"Verification result: predicted_user={predicted_user}, "
            f"confidence={confidence}, status={status}"
        )

        if str(predicted_user) != user_id:
            raise ValueError(
                f"Verification FAILED: expected ID '{user_id}' but got "
                f"'{predicted_user}' (confidence={confidence}). "
                f"The employee may need to re-register with better images."
            )

        logger.info(f"Verification PASSED for {username} (confidence={confidence})")

    except req.exceptions.ConnectionError:
        logger.warning(
            f"Could not reach API at {predict_url} — skipping live verification. "
            f"Database registration is still valid."
        )
    except req.exceptions.Timeout:
        logger.warning(
            f"API prediction timed out — skipping live verification."
        )

    return True


def push_to_feast(**context):
    """
    Push the extracted embedding to Feast Feature Store via HTTP API.
    This ensures the embedding is available in the online store (Redis)
    for low-latency inference immediately after onboarding.
    """
    import requests
    from datetime import datetime
    
    ti = context['ti']
    employee_data = ti.xcom_pull(task_ids='validate_employee_data', key='employee_data')
    embedding = ti.xcom_pull(task_ids='extract_embeddings', key='embedding')
    user_id = employee_data['user_id']
    username = employee_data['username']
    
    logger.info(f"Pushing embedding for {username} (ID: {user_id}) to Feast Feature Store")
    
    # Format payload for Feast Push API
    payload = {
        "push_source_name": "employee_embedding_push",
        "df": {
            "employee_id": [user_id],
            "embedding": [embedding],
            "embedding_model": ["facenet"],
            "embedding_version": [1],
            "image_quality_score": [1.0],
            "num_faces_detected": [1],
            "registration_confidence": [1.0],
            "event_timestamp": [datetime.utcnow().isoformat() + "Z"],
            "created_at": [datetime.utcnow().isoformat() + "Z"],
        }
    }
    
    try:
        url = f"{FEAST_SERVER_URL}/push"
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully pushed embedding for {username} to Feast")
    except Exception as e:
        logger.error(f"Failed to push to Feast: {e}")
        # We don't fail the DAG, as Postgres is the source of truth,
        # but we should log it clearly.
        
    return True


# Task definitions
# ═════════════════════════════════════════════════════════════════════════════

validate = PythonOperator(
    task_id='validate_employee_data',
    python_callable=validate_employee_data,
    provide_context=True,
    dag=dag,
)

detect_faces = PythonOperator(
    task_id='detect_validate_faces',
    python_callable=detect_and_validate_faces,
    provide_context=True,
    dag=dag,
)

extract = PythonOperator(
    task_id='extract_embeddings',
    python_callable=extract_embeddings,
    provide_context=True,
    dag=dag,
)

register = PythonOperator(
    task_id='register_in_database',
    python_callable=register_in_database,
    provide_context=True,
    dag=dag,
)

push_feast = PythonOperator(
    task_id='push_to_feast',
    python_callable=push_to_feast,
    provide_context=True,
    dag=dag,
)

verify = PythonOperator(
    task_id='verify_registration',
    python_callable=verify_registration,
    provide_context=True,
    dag=dag,
)


# Define task dependencies
# validate -> detect_faces -> extract_embeddings -> register -> push_feast -> verify
validate >> detect_faces >> extract >> register >> push_feast >> verify
