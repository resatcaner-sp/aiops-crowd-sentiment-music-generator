"""Event synchronization engine for aligning timestamps from different data sources."""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from pydantic import BaseModel, Field

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.services.synchronization.event_buffer import EventBuffer
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class TimestampSource(BaseModel):
    """Model for a timestamp source with priority.
    
    Attributes:
        name: Name of the timestamp source
        priority: Priority level (lower number means higher priority)
        offset: Offset from reference time in seconds
    """
    
    name: str
    priority: int
    offset: float = 0.0


class SyncEngine:
    """Synchronizes timestamps from different data sources.
    
    This engine aligns timestamps from different sources (data API, video feed)
    and provides methods for buffering and retrieving events.
    
    Attributes:
        kickoff_timestamp: Reference timestamp for synchronization
        source_offsets: Dictionary of timestamp sources with offsets
        event_buffer: Buffer for storing and retrieving events
        logger: Logger instance
    """
    
    def __init__(self, buffer_size: int = 60, cleanup_interval: int = 10) -> None:
        """Initialize synchronization engine with empty buffers.
        
        Args:
            buffer_size: Maximum age of events in seconds
            cleanup_interval: Interval between automatic cleanups in seconds
        """
        self.kickoff_timestamp: Optional[float] = None
        self.source_offsets: Dict[str, TimestampSource] = {}
        self.event_buffer = EventBuffer(buffer_size=buffer_size, cleanup_interval=cleanup_interval)
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    def register_timestamp_source(
        self, name: str, priority: int = 100, initial_offset: float = 0.0
    ) -> None:
        """Register a timestamp source with priority.
        
        Args:
            name: Name of the timestamp source
            priority: Priority level (lower number means higher priority)
            initial_offset: Initial offset from reference time in seconds
            
        Raises:
            SynchronizationError: If source already exists
        """
        if name in self.source_offsets:
            raise SynchronizationError(f"Timestamp source '{name}' already registered")
        
        self.source_offsets[name] = TimestampSource(
            name=name,
            priority=priority,
            offset=initial_offset
        )
        self.logger.info(f"Registered timestamp source '{name}' with priority {priority}")
    
    @with_error_handling
    def sync_with_kickoff(
        self, reference_source: str, reference_timestamp: float, 
        target_source: str, target_timestamp: float
    ) -> None:
        """Establish synchronization point at kick-off.
        
        Args:
            reference_source: Name of the reference timestamp source
            reference_timestamp: Reference timestamp value
            target_source: Name of the target timestamp source
            target_timestamp: Target timestamp value
            
        Raises:
            SynchronizationError: If sources are not registered
        """
        # Verify sources exist
        if reference_source not in self.source_offsets:
            raise SynchronizationError(f"Reference source '{reference_source}' not registered")
        
        if target_source not in self.source_offsets:
            raise SynchronizationError(f"Target source '{target_source}' not registered")
        
        # Calculate offset
        offset = target_timestamp - reference_timestamp
        
        # Store kickoff timestamp from reference source
        self.kickoff_timestamp = reference_timestamp
        
        # Update target source offset
        self.source_offsets[target_source].offset = offset
        
        self.logger.info(
            f"Synchronized '{target_source}' to '{reference_source}' "
            f"with offset {offset:.3f}s at kickoff"
        )
    
    @with_error_handling
    def align_timestamp(
        self, timestamp: float, source: str, target_source: str = "reference"
    ) -> float:
        """Convert timestamp from one source to another.
        
        Args:
            timestamp: Timestamp to convert
            source: Source of the timestamp
            target_source: Target source to convert to (default: reference)
            
        Returns:
            Aligned timestamp
            
        Raises:
            SynchronizationError: If sources are not registered or not synchronized
        """
        # Verify sources exist
        if source not in self.source_offsets:
            raise SynchronizationError(f"Source '{source}' not registered")
        
        if target_source != "reference" and target_source not in self.source_offsets:
            raise SynchronizationError(f"Target source '{target_source}' not registered")
        
        # Verify kickoff timestamp exists
        if self.kickoff_timestamp is None:
            raise SynchronizationError("No synchronization point established")
        
        # Get source offset
        source_offset = self.source_offsets[source].offset
        
        # Convert to reference time
        reference_time = timestamp - source_offset
        
        # If target is reference, return reference time
        if target_source == "reference":
            return reference_time
        
        # Otherwise, convert to target time
        target_offset = self.source_offsets[target_source].offset
        return reference_time + target_offset
    
    @with_error_handling
    def buffer_event(self, event: MatchEvent, source: str = "data_api") -> None:
        """Store event in buffer for later processing with audio.
        
        Args:
            event: Match event to buffer
            source: Source of the event
            
        Raises:
            SynchronizationError: If source is not registered or not synchronized
        """
        # Verify source exists
        if source not in self.source_offsets:
            raise SynchronizationError(f"Source '{source}' not registered")
        
        # Verify kickoff timestamp exists
        if self.kickoff_timestamp is None:
            raise SynchronizationError("No synchronization point established")
        
        # Align timestamp to reference timeline
        aligned_timestamp = self.align_timestamp(event.timestamp, source)
        
        # Add event to buffer
        self.event_buffer.add_event(event, aligned_timestamp, source)
        
        self.logger.debug(
            f"Buffered event {event.id} of type '{event.type}' "
            f"with aligned timestamp {aligned_timestamp:.3f}s"
        )
    
    @with_error_handling
    def get_events_for_audio(
        self, audio_timestamp: float, source: str = "hls", window: float = 5.0
    ) -> List[MatchEvent]:
        """Retrieve events that correspond to a specific audio segment.
        
        Args:
            audio_timestamp: Timestamp of the audio segment
            source: Source of the audio timestamp
            window: Time window in seconds to look for events
            
        Returns:
            List of match events in the time window
            
        Raises:
            SynchronizationError: If source is not registered or not synchronized
        """
        # Verify source exists
        if source not in self.source_offsets:
            raise SynchronizationError(f"Source '{source}' not registered")
        
        # Verify kickoff timestamp exists
        if self.kickoff_timestamp is None:
            raise SynchronizationError("No synchronization point established")
        
        # Align audio timestamp to reference timeline
        aligned_audio_timestamp = self.align_timestamp(audio_timestamp, source)
        
        # Calculate time window
        start_time = aligned_audio_timestamp - window
        end_time = aligned_audio_timestamp + window
        
        # Get events in time window
        events = self.event_buffer.get_events_in_window(start_time, end_time)
        
        self.logger.debug(
            f"Found {len(events)} events for audio at "
            f"{aligned_audio_timestamp:.3f}s (window: ±{window:.1f}s)"
        )
        
        return events
    
    @with_error_handling
    def get_events_by_type(
        self, event_type: str, audio_timestamp: Optional[float] = None,
        source: str = "hls", window: float = 5.0
    ) -> List[MatchEvent]:
        """Retrieve events of a specific type, optionally within a time window.
        
        Args:
            event_type: Type of events to retrieve
            audio_timestamp: Optional timestamp to center the time window
            source: Source of the audio timestamp
            window: Time window in seconds to look for events
            
        Returns:
            List of match events of the specified type
            
        Raises:
            SynchronizationError: If source is not registered or not synchronized
        """
        # If no audio timestamp, get all events of the specified type
        if audio_timestamp is None:
            return self.event_buffer.get_events_by_type(event_type)
        
        # Verify source exists
        if source not in self.source_offsets:
            raise SynchronizationError(f"Source '{source}' not registered")
        
        # Verify kickoff timestamp exists
        if self.kickoff_timestamp is None:
            raise SynchronizationError("No synchronization point established")
        
        # Align audio timestamp to reference timeline
        aligned_audio_timestamp = self.align_timestamp(audio_timestamp, source)
        
        # Calculate time window
        start_time = aligned_audio_timestamp - window
        end_time = aligned_audio_timestamp + window
        
        # Get events of the specified type in the time window
        events = self.event_buffer.get_events_by_type(
            event_type, (start_time, end_time)
        )
        
        self.logger.debug(
            f"Found {len(events)} events of type '{event_type}' for audio at "
            f"{aligned_audio_timestamp:.3f}s (window: ±{window:.1f}s)"
        )
        
        return events
    
    @with_error_handling
    def resolve_timestamp_conflict(
        self, timestamps: Dict[str, float]
    ) -> float:
        """Resolve conflicts between timestamps from different sources.
        
        Uses priority hierarchy to determine the most reliable timestamp.
        
        Args:
            timestamps: Dictionary of timestamps by source
            
        Returns:
            Resolved timestamp
            
        Raises:
            SynchronizationError: If no valid timestamps are provided
        """
        if not timestamps:
            raise SynchronizationError("No timestamps provided")
        
        # Filter sources that are registered
        valid_sources = {
            source: ts for source, ts in timestamps.items() 
            if source in self.source_offsets
        }
        
        if not valid_sources:
            raise SynchronizationError("No valid timestamp sources provided")
        
        # Sort sources by priority
        sorted_sources = sorted(
            valid_sources.keys(),
            key=lambda s: self.source_offsets[s].priority
        )
        
        # Use highest priority source
        highest_priority_source = sorted_sources[0]
        selected_timestamp = valid_sources[highest_priority_source]
        
        self.logger.debug(
            f"Resolved timestamp conflict using source '{highest_priority_source}' "
            f"with value {selected_timestamp:.3f}s"
        )
        
        return selected_timestamp
    
    @with_error_handling
    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        self.event_buffer.clear()
        self.logger.info("Event buffer cleared")
    
    @with_error_handling
    def set_buffer_size(self, seconds: int) -> None:
        """Set the buffer size in seconds.
        
        Args:
            seconds: Buffer size in seconds
            
        Raises:
            SynchronizationError: If seconds is not positive
        """
        self.event_buffer.set_buffer_size(seconds)
        self.logger.info(f"Buffer size set to {seconds} seconds")
    
    @with_error_handling
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the event buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = self.event_buffer.get_stats()
        
        # Add synchronization info
        stats["synchronization"] = {
            "kickoff_timestamp": self.kickoff_timestamp,
            "sources": len(self.source_offsets),
            "source_names": list(self.source_offsets.keys())
        }
        
        return stats