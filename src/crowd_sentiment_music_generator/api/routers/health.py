"""Health check router."""

from fastapi import APIRouter, Response, status

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "ok"}