"""Override router for manual control of music generation."""

import logging
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/override", tags=["override"])
logger = logging.getLogger(__name__)


class MusicOverride(BaseModel):
    """Music override model."""
    
    id: str
    match_id: str
    name: str
    description: Optional[str] = None
    audio_url: str
    duration: float
    tags: List[str] = []
    created_at: float


class OverrideRequest(BaseModel):
    """Override request model."""
    
    match_id: str
    segment_id: Optional[str] = None
    override_id: str
    reason: Optional[str] = None
    duration: Optional[float] = None
    fade_in: float = 0.5
    fade_out: float = 0.5


class OverrideResponse(BaseModel):
    """Override response model."""
    
    request_id: str
    match_id: str
    segment_id: Optional[str] = None
    override_id: str
    status: str = "applied"  # applied, pending, failed
    applied_at: float
    expires_at: Optional[float] = None


@router.get("/music", response_model=List[MusicOverride])
async def get_override_options(
    match_id: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
) -> List[MusicOverride]:
    """Get available music override options.
    
    Args:
        match_id: Filter by match ID (optional)
        tag: Filter by tag (optional)
        
    Returns:
        List of music override options
    """
    # In a real implementation, this would retrieve override options from a database
    # For this implementation, we'll return mock data
    overrides = [
        MusicOverride(
            id="override-1",
            match_id="match-123",
            name="Goal Celebration",
            description="Triumphant orchestral piece for goal celebrations",
            audio_url="https://example.com/overrides/goal-celebration.mp3",
            duration=30.0,
            tags=["goal", "celebration", "orchestral"],
            created_at=1626912000.0,  # Unix timestamp
        ),
        MusicOverride(
            id="override-2",
            match_id="match-123",
            name="Tension Builder",
            description="Suspenseful music for tense moments",
            audio_url="https://example.com/overrides/tension-builder.mp3",
            duration=45.0,
            tags=["tension", "suspense", "cinematic"],
            created_at=1626912600.0,  # Unix timestamp
        ),
        MusicOverride(
            id="override-3",
            match_id="",  # Empty match ID means it's available for all matches
            name="Generic Celebration",
            description="Generic celebration music for any match",
            audio_url="https://example.com/overrides/generic-celebration.mp3",
            duration=20.0,
            tags=["celebration", "generic", "electronic"],
            created_at=1626913200.0,  # Unix timestamp
        ),
    ]
    
    # Filter by match ID if provided
    if match_id:
        overrides = [o for o in overrides if o.match_id == match_id or o.match_id == ""]
    
    # Filter by tag if provided
    if tag:
        overrides = [o for o in overrides if tag in o.tags]
    
    return overrides


@router.post("/music", response_model=MusicOverride, status_code=status.HTTP_201_CREATED)
async def create_override_option(override: MusicOverride) -> MusicOverride:
    """Create music override option.
    
    Args:
        override: Music override option
        
    Returns:
        Created music override option
    """
    # In a real implementation, this would create an override option in a database
    # For this implementation, we'll just return the input
    return override


@router.post("/apply", response_model=OverrideResponse)
async def apply_override(request: OverrideRequest) -> OverrideResponse:
    """Apply music override.
    
    Args:
        request: Override request
        
    Returns:
        Override response
    """
    # In a real implementation, this would apply an override in the system
    # For this implementation, we'll return mock data
    import time
    import uuid
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Get current timestamp
    current_time = time.time()
    
    # Calculate expiration time if duration is provided
    expires_at = None
    if request.duration is not None:
        expires_at = current_time + request.duration
    
    return OverrideResponse(
        request_id=request_id,
        match_id=request.match_id,
        segment_id=request.segment_id,
        override_id=request.override_id,
        status="applied",
        applied_at=current_time,
        expires_at=expires_at,
    )


@router.get("/active", response_model=List[OverrideResponse])
async def get_active_overrides(
    match_id: Optional[str] = Query(None),
) -> List[OverrideResponse]:
    """Get active music overrides.
    
    Args:
        match_id: Filter by match ID (optional)
        
    Returns:
        List of active override responses
    """
    # In a real implementation, this would retrieve active overrides from a database
    # For this implementation, we'll return mock data
    import time
    
    # Get current timestamp
    current_time = time.time()
    
    overrides = [
        OverrideResponse(
            request_id="request-1",
            match_id="match-123",
            segment_id="segment-1",
            override_id="override-1",
            status="applied",
            applied_at=current_time - 60.0,  # 1 minute ago
            expires_at=current_time + 60.0,  # 1 minute from now
        ),
        OverrideResponse(
            request_id="request-2",
            match_id="match-456",
            segment_id=None,
            override_id="override-3",
            status="applied",
            applied_at=current_time - 120.0,  # 2 minutes ago
            expires_at=current_time + 120.0,  # 2 minutes from now
        ),
    ]
    
    # Filter by match ID if provided
    if match_id:
        overrides = [o for o in overrides if o.match_id == match_id]
    
    return overrides


@router.delete("/cancel/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_override(request_id: str) -> None:
    """Cancel music override.
    
    Args:
        request_id: Override request ID
    """
    # In a real implementation, this would cancel an override in the system
    # For this implementation, we'll just return
    pass