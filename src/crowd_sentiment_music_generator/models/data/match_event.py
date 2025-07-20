"""Match event data model."""

from typing import Dict, Optional

from pydantic import BaseModel


class MatchEvent(BaseModel):
    """Pydantic model for match events.
    
    Attributes:
        id: Unique identifier for the event
        type: Type of event (goal, card, penalty, etc.)
        timestamp: Time when the event occurred
        team_id: Identifier for the team involved in the event
        player_id: Optional identifier for the player involved
        position: Optional position data (x, y coordinates)
        additional_data: Optional additional event data
    """
    
    id: str
    type: str  # goal, card, penalty, etc.
    timestamp: float
    team_id: str
    player_id: Optional[str] = None
    position: Optional[Dict[str, float]] = None  # x, y coordinates
    additional_data: Optional[Dict[str, object]] = None