"""Music generation router."""

from fastapi import APIRouter, Depends, HTTPException, status

from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters

router = APIRouter(prefix="/music", tags=["music"])


@router.get("/parameters", response_model=MusicalParameters)
async def get_current_parameters() -> MusicalParameters:
    """Get current musical parameters.
    
    Returns:
        Current musical parameters
    """
    # TODO: Implement this
    return MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.5,
        instrumentation=["piano", "strings"],
        mood="neutral",
    )


@router.post("/parameters", response_model=MusicalParameters)
async def update_parameters(parameters: MusicalParameters) -> MusicalParameters:
    """Update musical parameters.
    
    Args:
        parameters: New musical parameters
        
    Returns:
        Updated musical parameters
    """
    # TODO: Implement this
    return parameters