"""Sync engine configuration model."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """Configuration for the synchronization engine.
    
    Attributes:
        buffer_size: Maximum size of the event buffer in seconds
        cleanup_interval: Interval for cleaning up old events in seconds
        data_source_priorities: Priority order for data sources (higher number = higher priority)
        kickoff_window: Time window for detecting kickoff events in seconds
        max_timestamp_drift: Maximum allowed drift between data sources in seconds
    """
    
    buffer_size: float = Field(default=60.0, description="Maximum size of the event buffer in seconds")
    cleanup_interval: float = Field(default=10.0, description="Interval for cleaning up old events in seconds")
    data_source_priorities: Dict[str, int] = Field(
        default={"data_api": 100, "video_feed": 50},
        description="Priority order for data sources (higher number = higher priority)"
    )
    kickoff_window: float = Field(default=5.0, description="Time window for detecting kickoff events in seconds")
    max_timestamp_drift: float = Field(default=1.0, description="Maximum allowed drift between data sources in seconds")