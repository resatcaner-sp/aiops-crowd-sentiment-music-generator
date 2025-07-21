"""Highlight segment data model."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HighlightSegment(BaseModel):
    """Pydantic model for video highlight segments.
    
    Attributes:
        id: Unique identifier for the segment
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
        key_moment_time: Timestamp of the key moment in the segment
        video_path: Path to the video file
        events: List of event IDs associated with this segment
        title: Optional title for the segment
        description: Optional description of the segment
        tags: Optional tags for the segment
        metadata: Optional additional metadata
    """
    
    id: str
    start_time: float
    end_time: float
    key_moment_time: float
    video_path: str
    events: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, object]] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment in seconds."""
        return self.end_time - self.start_time