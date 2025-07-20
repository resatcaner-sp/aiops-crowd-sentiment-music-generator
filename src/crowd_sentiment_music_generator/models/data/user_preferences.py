"""User preferences data model."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class UserPreferences(BaseModel):
    """Pydantic model for user preferences.
    
    Attributes:
        music_intensity: Preferred music intensity level (1-5 scale)
        preferred_genres: List of preferred music genres
        music_enabled: Whether music is enabled
        team_preferences: Team-specific preferences
        cultural_style: Optional cultural style preference
    """
    
    music_intensity: int = 3  # 1-5 scale
    preferred_genres: List[str] = ["orchestral"]
    music_enabled: bool = True
    team_preferences: Dict[str, Dict[str, object]] = {}
    cultural_style: Optional[str] = None