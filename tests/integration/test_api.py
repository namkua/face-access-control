"""
Test suite for Face Recognition API.
"""
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import app and dependencies
from api.app import app
from api.database import Base, get_db
from api.models import User

import os

# Test database URL
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/test_db"
)

# Create test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


import pytest_asyncio

@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Create test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """Create test client."""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


# ============================================================================
# User Management & Enrollment Tests
# ============================================================================

@pytest.mark.asyncio
async def test_create_user(client):
    """Test unified user enrollment with an image."""
    # Mock image bytes
    img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = [("files", ("face.png", img_data, "image/png"))]
    response = await client.post(
        "/api/v1/users/enroll",
        data={
            "email": "newuser@example.com",
            "username": "newuser",
            "full_name": "New User",
        },
        files=files
    )
    # The endpoint returns HTTP_202_ACCEPTED (202) for background processing
    assert response.status_code == 202
    data = response.json()
    assert "dag_run_id" in data
    assert "user_id" in data
    assert "onboarding pipeline triggered" in data["message"]


@pytest.mark.asyncio
async def test_create_duplicate_email(client, test_user):
    """Test creation with duplicate email."""
    img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = [("files", ("face.png", img_data, "image/png"))]
    response = await client.post(
        "/api/v1/users/enroll",
        data={
            "email": "test@example.com",  # Already exists
            "username": "anotheruser",
            "full_name": "Another User",
        },
        files=files
    )
    assert response.status_code == 400


# ============================================================================
# Face Recognition Tests
# ============================================================================

@pytest.mark.asyncio
async def test_predict_invalid_file_type(client):
    """Test prediction with invalid file type."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = await client.post(
        "/api/v1/predict",
        files=files
    )
    assert response.status_code == 400


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
