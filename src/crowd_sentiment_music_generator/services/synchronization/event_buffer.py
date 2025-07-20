"""Event buffer for storing and retrieving timestamped events."""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta

from pydantic import BaseModel

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class BufferedEvent(BaseModel):
    """Model for an event stored in the buffer.
    
    Attributes:
        event: The original match event
        aligned_timestamp: Timestamp aligned to reference timeline
        source: Source of the event
        processed: Whether the event has been processed
        created_at: Time when the event was added to the buffer
    """
    
    event: MatchEvent
    aligned_timestamp: float
    source: str
    processed: bool = False
    created_at: float = time.time()


class EventBuffer:
    """Buffer for storing and retrieving timestamped events.
    
    This class provides methods for storing events with timestamps,
    retrieving events for specific time windows, and cleaning up old events.
    
    Attributes:
        buffer: Dictionary of buffered events by ID
        buffer_size: Maximum age of events in seconds
        cleanup_interval: Interval between automatic cleanups in seconds
        last_cleanup: Timestamp of last cleanup
        logger: Logger instance
    """
    
    def __init__(self, buffer_size: int = 60, cleanup_interval: int = 10) -> None:
        """Initialize event buffer.
        
        Args:
            buffer_size: Maximum age of events in seconds
            cleanup_interval: Interval between automatic cleanups in seconds
        """
        self.buffer: Dict[str, BufferedEvent] = {}
        self.buffer_size = buffer_size
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    def add_event(
        self, event: MatchEvent, aligned_timestamp: float, source: str = "data_api"
    ) -> None:
        """Add an event to the buffer.
        
        Args:
            event: Match event to buffer
            aligned_timestamp: Timestamp aligned to reference timeline
            source: Source of the event
            
        Raises:
            SynchronizationError: If event with same ID already exists
        """
        if event.id in self.buffer:
            raise SynchronizationError(f"Event with ID '{event.id}' already exists in buffer")
        
        # Create buffered event
        buffered_event = BufferedEvent(
            event=event,
            aligned_timestamp=aligned_timestamp,
            source=source
        )
        
        # Add to buffer
        self.buffer[event.id] = buffered_event
        
        self.logger.debug(
            f"Added event {event.id} of type '{event.type}' "
            f"with aligned timestamp {aligned_timestamp:.3f}s to buffer"
        )
        
        # Check if cleanup is needed
        self._check_cleanup()
    
    @with_error_handling
    def update_event(self, event_id: str, processed: bool = True) -> None:
        """Update an event in the buffer.
        
        Args:
            event_id: ID of the event to update
            processed: Whether the event has been processed
            
        Raises:
            SynchronizationError: If event does not exist
        """
        if event_id not in self.buffer:
            raise SynchronizationError(f"Event with ID '{event_id}' not found in buffer")
        
        # Update event
        self.buffer[event_id].processed = processed
        
        self.logger.debug(f"Updated event {event_id} (processed={processed})")
    
    @with_error_handling
    def get_events_in_window(
        self, start_time: float, end_time: float, mark_processed: bool = True
    ) -> List[MatchEvent]:
        """Get events within a time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            mark_processed: Whether to mark retrieved events as processed
            
        Returns:
            List of match events in the time window
            
        Raises:
            SynchronizationError: If start_time is after end_time
        """
        if start_time > end_time:
            raise SynchronizationError("Start time must be before end time")
        
        # Find events in the time window
        matching_events = []
        matching_ids = []
        
        for event_id, buffered_event in self.buffer.items():
            event_timestamp = buffered_event.aligned_timestamp
            if start_time <= event_timestamp <= end_time:
                matching_events.append(buffered_event.event)
                matching_ids.append(event_id)
        
        # Mark events as processed if requested
        if mark_processed:
            for event_id in matching_ids:
                self.buffer[event_id].processed = True
        
        # Sort events by timestamp
        matching_events.sort(key=lambda e: e.timestamp)
        
        self.logger.debug(
            f"Found {len(matching_events)} events in window "
            f"[{start_time:.3f}s, {end_time:.3f}s]"
        )
        
        return matching_events
    
    @with_error_handling
    def get_events_by_type(
        self, event_type: str, time_window: Optional[Tuple[float, float]] = None,
        mark_processed: bool = False
    ) -> List[MatchEvent]:
        """Get events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            time_window: Optional time window (start_time, end_time)
            mark_processed: Whether to mark retrieved events as processed
            
        Returns:
            List of match events of the specified type
        """
        # Find events of the specified type
        matching_events = []
        matching_ids = []
        
        for event_id, buffered_event in self.buffer.items():
            # Check event type
            if buffered_event.event.type != event_type:
                continue
            
            # Check time window if specified
            if time_window is not None:
                start_time, end_time = time_window
                event_timestamp = buffered_event.aligned_timestamp
                if not (start_time <= event_timestamp <= end_time):
                    continue
            
            matching_events.append(buffered_event.event)
            matching_ids.append(event_id)
        
        # Mark events as processed if requested
        if mark_processed:
            for event_id in matching_ids:
                self.buffer[event_id].processed = True
        
        # Sort events by timestamp
        matching_events.sort(key=lambda e: e.timestamp)
        
        self.logger.debug(
            f"Found {len(matching_events)} events of type '{event_type}'"
        )
        
        return matching_events
    
    @with_error_handling
    def get_latest_events(
        self, count: int = 10, mark_processed: bool = False
    ) -> List[MatchEvent]:
        """Get the latest events in the buffer.
        
        Args:
            count: Maximum number of events to retrieve
            mark_processed: Whether to mark retrieved events as processed
            
        Returns:
            List of latest match events
        """
        # Sort events by timestamp
        sorted_events = sorted(
            self.buffer.items(),
            key=lambda item: item[1].aligned_timestamp,
            reverse=True
        )
        
        # Get latest events
        latest_events = []
        latest_ids = []
        
        for i, (event_id, buffered_event) in enumerate(sorted_events):
            if i >= count:
                break
            
            latest_events.append(buffered_event.event)
            latest_ids.append(event_id)
        
        # Mark events as processed if requested
        if mark_processed:
            for event_id in latest_ids:
                self.buffer[event_id].processed = True
        
        # Sort events by timestamp (ascending)
        latest_events.sort(key=lambda e: e.timestamp)
        
        self.logger.debug(f"Retrieved {len(latest_events)} latest events")
        
        return latest_events
    
    @with_error_handling
    def clear(self) -> None:
        """Clear the event buffer."""
        self.buffer.clear()
        self.logger.info("Event buffer cleared")
    
    @with_error_handling
    def set_buffer_size(self, seconds: int) -> None:
        """Set the buffer size in seconds.
        
        Args:
            seconds: Buffer size in seconds
            
        Raises:
            SynchronizationError: If seconds is not positive
        """
        if seconds <= 0:
            raise SynchronizationError("Buffer size must be positive")
        
        self.buffer_size = seconds
        self.logger.info(f"Buffer size set to {seconds} seconds")
        
        # Clean up buffer with new size
        self.cleanup()
    
    @with_error_handling
    def cleanup(self) -> int:
        """Clean up old events from the buffer.
        
        Returns:
            Number of events removed
        """
        if not self.buffer:
            return 0
        
        # Calculate cutoff time
        cutoff_time = time.time() - self.buffer_size
        
        # Find old events
        old_event_ids = []
        
        for event_id, buffered_event in self.buffer.items():
            # Remove if:
            # 1. Event is older than buffer_size AND has been processed, OR
            # 2. Event is MUCH older than buffer_size (2x) regardless of processing
            if ((buffered_event.created_at < cutoff_time and buffered_event.processed) or
                    buffered_event.created_at < cutoff_time - self.buffer_size):
                old_event_ids.append(event_id)
        
        # Remove old events
        for event_id in old_event_ids:
            del self.buffer[event_id]
        
        # Update last cleanup time
        self.last_cleanup = time.time()
        
        if old_event_ids:
            self.logger.debug(f"Removed {len(old_event_ids)} old events from buffer")
        
        return len(old_event_ids)
    
    def _check_cleanup(self) -> None:
        """Check if cleanup is needed and perform if necessary."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
    
    @with_error_handling
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        # Count events by type
        event_types: Dict[str, int] = {}
        processed_count = 0
        unprocessed_count = 0
        
        for buffered_event in self.buffer.values():
            event_type = buffered_event.event.type
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if buffered_event.processed:
                processed_count += 1
            else:
                unprocessed_count += 1
        
        # Calculate age statistics
        current_time = time.time()
        ages = [current_time - be.created_at for be in self.buffer.values()]
        
        avg_age = sum(ages) / len(ages) if ages else 0
        max_age = max(ages) if ages else 0
        
        return {
            "total_events": len(self.buffer),
            "processed_events": processed_count,
            "unprocessed_events": unprocessed_count,
            "event_types": event_types,
            "average_age": avg_age,
            "max_age": max_age,
            "buffer_size": self.buffer_size,
            "cleanup_interval": self.cleanup_interval,
        }