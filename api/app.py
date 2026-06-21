"""
Face Recognition MLOps API Service.

Production-ready FastAPI application with face recognition,
monitoring, and data pipeline integration.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List
import uuid as uuid_lib
import io
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Info
from starlette.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog
import numpy as np

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from api.config import settings
from api.database import get_db, init_db, close_db
from api.models import User, Prediction
from api.schemas import PredictionResponse, HealthResponse, ErrorResponse
from api.services.face_recognition import face_recognition_service
from api.services.kafka_producer import kafka_producer
from api.services.minio_client import minio_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
FACE_DETECTION_COUNT = Counter('face_detections_total', 'Total face detections', ['detected'])
APP_INFO = Info('app', 'Application information')

# Set app info
APP_INFO.info({
    'version': settings.app_version,
    'environment': settings.environment,
    'model': settings.model_name
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("application_startup", version=settings.app_version)
    
    try:
        # Initialize database
        await init_db()
        
        # Start Kafka producer
        await kafka_producer.start()

        # ── Warm-up inference backend ──────────────────────────────────
        # Eagerly try to connect to Triton so the first real request
        # doesn't pay the connection overhead.
        from api.services.face_recognition import _init_triton, _get_mtcnn, _get_facenet
        triton_client = _init_triton()
        if triton_client:
            logger.info("triton_warm_up_complete", url=settings.triton_url)
        else:
            # Triton not available → pre-load local MTCNN so the first
            # inference doesn't block while loading a 200 MB model.
            logger.info("triton_unavailable_warming_up_local_mtcnn")
            _get_mtcnn()   # loads MTCNN into RAM now
            _get_facenet()
        
        # ── Warm-up Feast Client ───────────────────────────────────────
        try:
            logger.info("warming_up_feast_client")
            from features.feast_client import get_feast_client
            get_feast_client()
            logger.info("feast_warm_up_complete")
        except Exception as e:
            logger.warning("feast_warm_up_failed", error=str(e))
        
        logger.info("application_ready")
        
    except Exception as e:
        logger.error("startup_error", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("application_shutdown")
    
    try:
        # Stop Kafka producer
        await kafka_producer.stop()
        
        # Close database
        await close_db()
        
        logger.info("shutdown_complete")
        
    except Exception as e:
        logger.error("shutdown_error", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="Face Recognition MLOps API",
    description="Production-ready face recognition system with MLOps",
    version=settings.app_version,
    lifespan=lifespan
)

# Configure OpenTelemetry Tracing
resource = Resource(attributes={
    SERVICE_NAME: "face-recognition-api"
})
tracer_provider = TracerProvider(resource=resource)
# Jaeger OTLP Exporter
jaeger_endpoint = f"http://{settings.jaeger_agent_host}:4317"
otlp_exporter = OTLPSpanExporter(endpoint=jaeger_endpoint, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for collecting metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    
    return response


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        dependencies={
            "database": "connected",
            "kafka": "connected",
            "minio": "connected",
        }
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Face Recognition Endpoints
# ============================================================================

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_face(
    request: Request,
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict identity from face image.
    
    - **file**: Image file containing a face
    - **threshold**: Optional custom similarity threshold (0.0-1.0)
    """
    start_time = time.time()
    prediction_uuid = str(uuid_lib.uuid4())
    
    try:
        # Validate file
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only JPEG and PNG are supported."
            )
        
        # Read file
        image_bytes = await file.read()
        
        # Check file size
        if len(image_bytes) > settings.max_image_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {settings.max_image_size_mb}MB."
            )
        
        # Detect face
        face_detected, bounding_box = await face_recognition_service.detect_face(image_bytes)
        FACE_DETECTION_COUNT.labels(detected=str(face_detected)).inc()
        
        if not face_detected:
            processing_time = (time.time() - start_time) * 1000
            PREDICTION_COUNT.labels(status="no_face").inc()
            
            await kafka_producer.send_access_log(
                user_id="anonymous",
                action="prediction",
                metadata={
                    "prediction_id": prediction_uuid,
                    "matched_user": None,
                    "confidence": None,
                    "processing_time_ms": processing_time,
                    "source_ip": request.client.host if request.client else "unknown",
                    "prediction_status": "no_face"
                }
            )
            
            return PredictionResponse(
                prediction_id=prediction_uuid,
                predicted_user=None,
                confidence=None,
                face_detected=False,
                processing_time_ms=processing_time,
                status="no_face",
                message="No face detected in image"
            )
        
        # Extract embedding
        embedding = await face_recognition_service.extract_embedding(image_bytes)
        
        if embedding is None:
            processing_time = (time.time() - start_time) * 1000
            PREDICTION_COUNT.labels(status="error").inc()
            
            await kafka_producer.send_access_log(
                user_id="anonymous",
                action="prediction",
                metadata={
                    "prediction_id": prediction_uuid,
                    "matched_user": None,
                    "confidence": None,
                    "processing_time_ms": processing_time,
                    "source_ip": request.client.host if request.client else "unknown",
                    "prediction_status": "error"
                }
            )
            
            return PredictionResponse(
                prediction_id=prediction_uuid,
                predicted_user=None,
                confidence=None,
                face_detected=True,
                processing_time_ms=processing_time,
                status="error",
                message="Failed to extract face embedding"
            )
        
        # Get all users with embeddings (to get active user list)
        result = await db.execute(select(User).where(User.face_embedding.isnot(None)))
        users = result.scalars().all()
        
        try:
            from features.feast_client import get_feast_client
            feast_client = get_feast_client()
            user_ids = [str(user.id) for user in users]
            # Try to read embeddings from Feast Redis online store
            database_embeddings = await feast_client.get_embeddings_for_matching(user_ids)
            if not database_embeddings:
                raise ValueError("No embeddings returned from Feast")
            logger.info("embeddings_loaded_from_feast", num_embeddings=len(database_embeddings))
        except Exception as e:
            logger.warning("feast_read_failed_falling_back_to_postgres", error=str(e))
            # Fallback to Postgres
            database_embeddings = {
                str(user.id): np.array(user.face_embedding)
                for user in users
            }
        
        # Find best match
        custom_threshold = threshold or settings.model_threshold
        matched_user, confidence = await face_recognition_service.find_best_match(
            query_embedding=embedding,
            database_embeddings=database_embeddings,
            threshold=custom_threshold
        )
        
        # Upload image to MinIO
        object_name = f"predictions/Year={datetime.utcnow().strftime('%Y')}/Month={datetime.utcnow().strftime('%m')}/{datetime.utcnow().strftime('%d')}/{prediction_uuid}.jpg"
        await minio_service.upload_file(
            file_bytes=image_bytes,
            object_name=object_name,
            metadata={
                "prediction_id": prediction_uuid,
                "matched_user": matched_user or "unknown"
            }
        )
        
        # Save prediction to database
        processing_time = (time.time() - start_time) * 1000
        prediction = Prediction(
            uuid=prediction_uuid,
            image_path=object_name,
            user_id=int(matched_user) if matched_user else None,
            confidence=confidence,
            processing_time_ms=processing_time,
            status="success" if matched_user else "no_match",
            meta_data={"bounding_box": bounding_box}
        )
        
        db.add(prediction)
        await db.commit()
        
        # Send to Kafka
        await kafka_producer.send_access_log(
            user_id=matched_user or "anonymous",
            action="prediction",
            metadata={
                "prediction_id": prediction_uuid,
                "matched_user": matched_user,
                "confidence": confidence,
                "processing_time_ms": processing_time,
                "source_ip": request.client.host if request.client else "unknown",
                "prediction_status": "success" if matched_user else "no_match"
            }
        )
        
        # Metrics
        PREDICTION_COUNT.labels(status="success" if matched_user else "no_match").inc()
        PREDICTION_LATENCY.observe(processing_time / 1000)
        
        return PredictionResponse(
            prediction_id=prediction_uuid,
            predicted_user=matched_user,
            confidence=confidence,
            face_detected=True,
            processing_time_ms=processing_time,
            status="success" if matched_user else "no_match",
            message=f"Match found: {matched_user}" if matched_user else "No match found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prediction_error", error=str(e), prediction_id=prediction_uuid)
        processing_time = (time.time() - start_time) * 1000
        PREDICTION_COUNT.labels(status="error").inc()
        
        await kafka_producer.send_access_log(
            user_id="anonymous",
            action="prediction",
            metadata={
                "prediction_id": prediction_uuid,
                "matched_user": None,
                "confidence": None,
                "processing_time_ms": processing_time,
                "source_ip": request.client.host if request.client else "unknown",
                "prediction_status": "error",
                "error": str(e)
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post("/api/v1/users/enroll", status_code=status.HTTP_202_ACCEPTED)
async def enroll_user(
    email: str = Form(...),
    username: str = Form(...),
    full_name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new employee and their face in one step.
    
    - **email**: Employee email
    - **username**: Unique username
    - **full_name**: Employee full name
    - **files**: List of image files containing the user's face (at least 1, recommended >=3)
    """
    
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file is required"
        )
        
    # Check if user already exists
    result = await db.execute(select(User).where((User.username == username) | (User.email == email)))
    user = result.scalar_one_or_none()
    
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create user
    user = User(
        email=email,
        username=username,
        full_name=full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info("user_created", user_id=user.id, username=user.username)
    
    # Send event to Kafka
    await kafka_producer.send_event("user_created", {
        "user_id": user.id,
        "username": user.username,
        "email": user.email
    })
    
    image_paths = []
    
    for idx, file in enumerate(files):
        # Read file
        image_bytes = await file.read()
        
        # Determine object name
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        object_name = f"onboarding/{user.username}/{user.username}_{idx}.{file_ext}"
        
        # Upload image to MinIO
        await minio_service.upload_file(
            file_bytes=image_bytes,
            object_name=object_name,
            content_type=file.content_type or "image/jpeg"
        )
        
        # Append S3 URI
        image_paths.append(f"s3://{minio_service.bucket_raw}/{object_name}")
        
    # Trigger Airflow DAG
    airflow_url = "http://airflow:8080/api/v1/dags/employee_onboarding/dagRuns"
    auth = ("admin", "admin")
    
    conf = {
        "user_id": str(user.id),
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "image_paths": image_paths
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                airflow_url,
                auth=auth,
                json={"conf": conf}
            )
            response.raise_for_status()
            dag_run_info = response.json()
            dag_run_id = dag_run_info.get("dag_run_id")
    except Exception as e:
        logger.error("airflow_trigger_failed", error=str(e), username=user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger onboarding pipeline: {str(e)}"
        )
        
    logger.info("onboarding_pipeline_triggered", username=user.username, dag_run_id=dag_run_id)
    
    # Send event
    await kafka_producer.send_event("face_registration_triggered", {
        "user_id": user.id,
        "username": user.username,
        "dag_run_id": dag_run_id
    })
    
    return {
        "message": f"User enrolled and onboarding pipeline triggered for {user.username}",
        "user_id": user.id,
        "dag_run_id": dag_run_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers
    )