"""Event handling router."""

from fastapi import APIRouter, HTTPException, status

from crowd_sentiment_music_generator.models.data.match_event import MatchEvent

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/", status_code=status.HTTP_201_CREATED)
async def process_event(event: MatchEvent) -> dict:
    """Process a match event.
    
    Args:
        event: The match event to process
        
    Returns:
        Processing status
    """
    # TODO: Implement this
    return {"status": "processed", "event_id": event.id}