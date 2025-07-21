"""Highlight music data model."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters


class MusicSegment(BaseModel):
    """Pydantic model for a music segment within a highlight.
    
    Attributes:
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
        parameters: Musical parameters for this segment
        transition_in: Whether to create a transition into this segment
        transition_out: Whether to create a transition out of this segment
        accent_time: Optional time for a musical accent within the segment
        accent_type: Optional type of accent to apply
    """
    
    start_time: float
    end_time: float
    parameters: MusicalParameters
    transition_in: bool = True
    transition_out: bool = True
    accent_time: Optional[float] = None
    accent_type: Optional[str] = None


class HighlightMusic(BaseModel):
    """Pydantic model for highlight music composition.
    
    Attributes:
        highlight_id: ID of the highlight this music is for
        segments: List of music segments that make up the composition
        base_parameters: Base musical parameters for the entire composition
        duration: Total duration of the music in seconds
        metadata: Optional additional metadata
    """
    
    highlight_id: str
    segments: List[MusicSegment]
    base_parameters: MusicalParameters
    duration: float
    metadata: Optional[Dict[str, object]] = None