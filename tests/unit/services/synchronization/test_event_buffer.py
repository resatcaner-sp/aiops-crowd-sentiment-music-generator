"""Unit tests for event buffer."""

import time
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.services.synchronization.event_buffer import (
    EventBuffer,
    BufferedEvent,
)


class TestEventBuffer:
    """Test cases for EventBuffer class."""
    
    @pytest.fixture
    def buffer(self) -> EventBuffer:
        """Create an EventBuffer instance for testing."""
        return EventBuffer(buffer_size=60, cleanup_interval=10)
    
    @pytest.fixture
    def sample_match_event(self) -> MatchEvent:
        """Create a sample match event."""
        return MatchEvent(
            id="123",
            type="goal",
            timestamp=1625097600.0,
            team_id="team1",
            player_id="player1",
            position={"x": 10.5, "y": 20.3},
            additional_data={"speed": 25.6, "angle": 45.0},
        )
    
    def test_initialization(self, buffer: EventBuffer) -> None:
        """Test buffer initialization."""
        assert buffer.buffer == {}
        assert buffer.buffer_size == 60
        assert buffer.cleanup_interval == 10
        assert buffer.last_cleanup <= time.time()
    
    def test_add_event(self, buffer: EventBuffer, sample_match_event: MatchEvent) -> None:
        """Test adding events to buffer."""
        # Add event
        buffer.add_event(sample_match_event, 1000.0, "data_api")
        
        # Verify buffer
        assert sample_match_event.id in buffer.buffer
        buffered_event = buffer.buffer[sample_match_event.id]
        assert buffered_event.event == sample_match_event
        assert buffered_event.aligned_timestamp == 1000.0
        assert buffered_event.source == "data_api"
        assert not buffered_event.processed
        
        # Try to add duplicate event
        with pytest.raises(SynchronizationError) as excinfo:
            buffer.add_event(sample_match_event, 1000.0, "data_api")
        
        # Verify error message
        assert "already exists in buffer" in str(excinfo.value)
    
    def test_update_event(self, buffer: EventBuffer, sample_match_event: MatchEvent) -> None:
        """Test updating events in buffer."""
        # Add event
        buffer.add_event(sample_match_event, 1000.0, "data_api")
        
        # Update event
        buffer.update_event(sample_match_event.id, processed=True)
        
        # Verify event was updated
        assert buffer.buffer[sample_match_event.id].processed
        
        # Try to update non-existent event
        with pytest.raises(SynchronizationError) as excinfo:
            buffer.update_event("non_existent", processed=True)
        
        # Verify error message
        assert "not found in buffer" in str(excinfo.value)
    
    def test_get_events_in_window(self, buffer: EventBuffer) -> None:
        """Test getting events in time window."""
        # Create events with different timestamps
        event1 = MatchEvent(
            id="123",
            type="goal",
            timestamp=1000.0,
            team_id="team1"
        )
        
        event2 = MatchEvent(
            id="456",
            type="card",
            timestamp=1010.0,
            team_id="team2"
        )
        
        event3 = MatchEvent(
            id="789",
            type="foul",
            timestamp=1020.0,
            team_id="team1"
        )
        
        # Add events to buffer
        buffer.add_event(event1, 1000.0, "data_api")
        buffer.add_event(event2, 1010.0, "data_api")
        buffer.add_event(event3, 1020.0, "data_api")
        
        # Get events in window
        events = buffer.get_events_in_window(995.0, 1015.0)
        
        # Verify events
        assert len(events) == 2
        assert events[0].id == event1.id
        assert events[1].id == event2.id
        
        # Verify events are marked as processed
        assert buffer.buffer[event1.id].processed
        assert buffer.buffer[event2.id].processed
        assert not buffer.buffer[event3.id].processed
        
        # Get events without marking as processed
        events = buffer.get_events_in_window(1015.0, 1025.0, mark_processed=False)
        
        # Verify events
        assert len(events) == 1
        assert events[0].id == event3.id
        
        # Verify event is not marked as processed
        assert not buffer.buffer[event3.id].processed
        
        # Try to get events with invalid window
        with pytest.raises(SynchronizationError) as excinfo:
            buffer.get_events_in_window(1015.0, 1005.0)
        
        # Verify error message
        assert "Start time must be before end time" in str(excinfo.value)
    
    def test_get_events_by_type(self, buffer: EventBuffer) -> None:
        """Test getting events by type."""
        # Create events with different types
        event1 = MatchEvent(
            id="123",
            type="goal",
            timestamp=1000.0,
            team_id="team1"
        )
        
        event2 = MatchEvent(
            id="456",
            type="card",
            timestamp=1010.0,
            team_id="team2"
        )
        
        event3 = MatchEvent(
            id="789",
            type="goal",
            timestamp=1020.0,
            team_id="team1"
        )
        
        # Add events to buffer
        buffer.add_event(event1, 1000.0, "data_api")
        buffer.add_event(event2, 1010.0, "data_api")
        buffer.add_event(event3, 1020.0, "data_api")
        
        # Get events by type
        events = buffer.get_events_by_type("goal")
        
        # Verify events
        assert len(events) == 2
        assert events[0].id == event1.id
        assert events[1].id == event3.id
        
        # Verify events are not marked as processed
        assert not buffer.buffer[event1.id].processed
        assert not buffer.buffer[event3.id].processed
        
        # Get events by type with time window
        events = buffer.get_events_by_type("goal", (1015.0, 1025.0), mark_processed=True)
        
        # Verify events
        assert len(events) == 1
        assert events[0].id == event3.id
        
        # Verify event is marked as processed
        assert not buffer.buffer[event1.id].processed
        assert buffer.buffer[event3.id].processed
    
    def test_get_latest_events(self, buffer: EventBuffer) -> None:
        """Test getting latest events."""
        # Create events with different timestamps
        event1 = MatchEvent(
            id="123",
            type="goal",
            timestamp=1000.0,
            team_id="team1"
        )
        
        event2 = MatchEvent(
            id="456",
            type="card",
            timestamp=1010.0,
            team_id="team2"
        )
        
        event3 = MatchEvent(
            id="789",
            type="foul",
            timestamp=1020.0,
            team_id="team1"
        )
        
        # Add events to buffer
        buffer.add_event(event1, 1000.0, "data_api")
        buffer.add_event(event2, 1010.0, "data_api")
        buffer.add_event(event3, 1020.0, "data_api")
        
        # Get latest events (limit 2)
        events = buffer.get_latest_events(2)
        
        # Verify events (should be sorted by timestamp)
        assert len(events) == 2
        assert events[0].id == event2.id
        assert events[1].id == event3.id
        
        # Verify events are not marked as processed
        assert not buffer.buffer[event2.id].processed
        assert not buffer.buffer[event3.id].processed
        
        # Get all latest events with mark_processed=True
        events = buffer.get_latest_events(10, mark_processed=True)
        
        # Verify events
        assert len(events) == 3
        assert events[0].id == event1.id
        assert events[1].id == event2.id
        assert events[2].id == event3.id
        
        # Verify events are marked as processed
        assert buffer.buffer[event1.id].processed
        assert buffer.buffer[event2.id].processed
        assert buffer.buffer[event3.id].processed
    
    def test_clear(self, buffer: EventBuffer, sample_match_event: MatchEvent) -> None:
        """Test clearing buffer."""
        # Add event
        buffer.add_event(sample_match_event, 1000.0, "data_api")
        
        # Verify buffer
        assert len(buffer.buffer) == 1
        
        # Clear buffer
        buffer.clear()
        
        # Verify buffer is empty
        assert len(buffer.buffer) == 0
    
    def test_set_buffer_size(self, buffer: EventBuffer) -> None:
        """Test setting buffer size."""
        # Set buffer size
        buffer.set_buffer_size(120)
        
        # Verify buffer size
        assert buffer.buffer_size == 120
        
        # Try to set invalid buffer size
        with pytest.raises(SynchronizationError) as excinfo:
            buffer.set_buffer_size(0)
        
        # Verify error message
        assert "Buffer size must be positive" in str(excinfo.value)
        
        with pytest.raises(SynchronizationError) as excinfo:
            buffer.set_buffer_size(-10)
        
        # Verify error message
        assert "Buffer size must be positive" in str(excinfo.value)
    
    def test_cleanup(self, buffer: EventBuffer) -> None:
        """Test cleaning up buffer."""
        # Mock time.time to return a fixed value
        current_time = 1000.0
        with patch("time.time", return_value=current_time):
            # Create events with different creation times
            event1 = MatchEvent(
                id="123",
                type="goal",
                timestamp=900.0,
                team_id="team1"
            )
            
            event2 = MatchEvent(
                id="456",
                type="card",
                timestamp=950.0,
                team_id="team2"
            )
            
            # Add events to buffer
            buffer.add_event(event1, 900.0, "data_api")
            buffer.add_event(event2, 950.0, "data_api")
            
            # Manually set created_at to simulate old events
            buffer.buffer["123"].created_at = current_time - 70  # Older than buffer_size
            buffer.buffer["456"].created_at = current_time - 30  # Within buffer_size
            
            # Mark event1 as processed
            buffer.buffer["123"].processed = True
            
            # Clean up buffer
            removed = buffer.cleanup()
            
            # Verify old processed event was removed
            assert removed == 1
            assert "123" not in buffer.buffer
            assert "456" in buffer.buffer
            
            # Add another event that's very old (2x buffer_size)
            event3 = MatchEvent(
                id="789",
                type="foul",
                timestamp=800.0,
                team_id="team1"
            )
            
            buffer.add_event(event3, 800.0, "data_api")
            
            # Manually set created_at to simulate very old event
            buffer.buffer["789"].created_at = current_time - 130  # Much older than buffer_size
            
            # Clean up buffer
            removed = buffer.cleanup()
            
            # Verify very old event was removed even though not processed
            assert removed == 1
            assert "789" not in buffer.buffer
    
    def test_check_cleanup(self, buffer: EventBuffer) -> None:
        """Test automatic cleanup check."""
        # Mock cleanup method
        with patch.object(buffer, "cleanup") as mock_cleanup:
            # Set last_cleanup to trigger cleanup
            buffer.last_cleanup = time.time() - buffer.cleanup_interval - 1
            
            # Add event to trigger _check_cleanup
            event = MatchEvent(
                id="123",
                type="goal",
                timestamp=1000.0,
                team_id="team1"
            )
            
            buffer.add_event(event, 1000.0, "data_api")
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once()
    
    def test_get_stats(self, buffer: EventBuffer) -> None:
        """Test getting buffer statistics."""
        # Create events with different types
        event1 = MatchEvent(
            id="123",
            type="goal",
            timestamp=1000.0,
            team_id="team1"
        )
        
        event2 = MatchEvent(
            id="456",
            type="card",
            timestamp=1010.0,
            team_id="team2"
        )
        
        event3 = MatchEvent(
            id="789",
            type="goal",
            timestamp=1020.0,
            team_id="team1"
        )
        
        # Add events to buffer
        buffer.add_event(event1, 1000.0, "data_api")
        buffer.add_event(event2, 1010.0, "data_api")
        buffer.add_event(event3, 1020.0, "data_api")
        
        # Mark some events as processed
        buffer.update_event("123", processed=True)
        
        # Get stats
        stats = buffer.get_stats()
        
        # Verify stats
        assert stats["total_events"] == 3
        assert stats["processed_events"] == 1
        assert stats["unprocessed_events"] == 2
        assert stats["event_types"] == {"goal": 2, "card": 1}
        assert stats["buffer_size"] == 60
        assert stats["cleanup_interval"] == 10
        assert "average_age" in stats
        assert "max_age" in stats
    
    @patch("crowd_sentiment_music_generator.services.synchronization.event_buffer.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, buffer: EventBuffer) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(buffer.add_event, "__wrapped__")
        assert hasattr(buffer.update_event, "__wrapped__")
        assert hasattr(buffer.get_events_in_window, "__wrapped__")
        assert hasattr(buffer.get_events_by_type, "__wrapped__")
        assert hasattr(buffer.get_latest_events, "__wrapped__")
        assert hasattr(buffer.clear, "__wrapped__")
        assert hasattr(buffer.set_buffer_size, "__wrapped__")
        assert hasattr(buffer.cleanup, "__wrapped__")
        assert hasattr(buffer.get_stats, "__wrapped__")