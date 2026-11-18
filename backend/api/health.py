
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Check whether the backend application is running.
    """
    return {
        "status": "healthy",
        "service": "Insurance Claims Intelligence API",
        "version": "1.0.0",
    }
