"""Musical parameters data model."""

from typing import List, Optional

from pydantic import BaseModel


class MusicalParameters(BaseModel):
    """Pydantic model for musical parameters.
    
    Attributes:
        tempo: Beats per minute
        key: Musical key (e.g., C Major, A Minor)
        intensity: Intensity level on a scale of 0-1
        instrumentation: List of instruments to use
        mood: Mood descriptor (bright, dark, tense, etc.)
        transition_duration: Optional duration for transitions between musical states
    """
    
    tempo: float  # BPM
    key: str  # C Major, A Minor, etc.
    intensity: float  # 0-1 scale
    instrumentation: List[str]
    mood: str  # bright, dark, tense, etc.
    transition_duration: Optional[float] = None