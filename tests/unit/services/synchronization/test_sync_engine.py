"""Unit tests for synchronization engine."""

import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.services.synchronization.sync_engine import (
    SyncEngine,
    TimestampSource,
)
from crowd_sentiment_music_generator.services.synchronization.event_buffer import EventBuffer


class TestSyncEngine:
    """Test cases for SyncEngine class."""
    
    @pytest.fixture
    def engine(self) -> SyncEngine:
        """Create a SyncEngine instance for testing."""
        return SyncEngine(buffer_size=60, cleanup_interval=10)
    
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
    
    def test_initialization(self, engine: SyncEngine) -> None:
        """Test engine initialization."""
        assert engine.kickoff_timestamp is None
        assert engine.source_offsets == {}
        assert isinstance(engine.event_buffer, EventBuffer)
        assert engine.event_buffer.buffer_size == 60
        assert engine.event_buffer.cleanup_interval == 10
    
    def test_register_timestamp_source(self, engine: SyncEngine) -> None:
        """Test registering timestamp sources."""
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2, 10.0)
        
        # Verify sources
        assert "data_api" in engine.source_offsets
        assert "hls" in engine.source_offsets
        assert engine.source_offsets["data_api"].priority == 1
        assert engine.source_offsets["data_api"].offset == 0.0
        assert engine.source_offsets["hls"].priority == 2
        assert engine.source_offsets["hls"].offset == 10.0
        
        # Try to register duplicate source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.register_timestamp_source("data_api", 3)
        
        # Verify error message
        assert "already registered" in str(excinfo.value)
    
    def test_sync_with_kickoff(self, engine: SyncEngine) -> None:
        """Test synchronizing with kickoff."""
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2)
        
        # Sync with kickoff
        engine.sync_with_kickoff("data_api", 1000.0, "hls", 1010.0)
        
        # Verify kickoff timestamp and offset
        assert engine.kickoff_timestamp == 1000.0
        assert engine.source_offsets["hls"].offset == 10.0
        
        # Try to sync with unregistered source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.sync_with_kickoff("unknown", 1000.0, "hls", 1010.0)
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine.sync_with_kickoff("data_api", 1000.0, "unknown", 1010.0)
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
    
    def test_align_timestamp(self, engine: SyncEngine) -> None:
        """Test aligning timestamps."""
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2)
        
        # Sync with kickoff
        engine.sync_with_kickoff("data_api", 1000.0, "hls", 1010.0)
        
        # Align timestamps
        assert engine.align_timestamp(1020.0, "data_api") == 1020.0  # To reference
        assert engine.align_timestamp(1030.0, "hls") == 1020.0  # To reference
        assert engine.align_timestamp(1020.0, "data_api", "hls") == 1030.0  # data_api to hls
        
        # Try to align with unregistered source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.align_timestamp(1020.0, "unknown")
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine.align_timestamp(1020.0, "data_api", "unknown")
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        # Try to align without synchronization
        engine_unsync = SyncEngine()
        engine_unsync.register_timestamp_source("data_api", 1)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine_unsync.align_timestamp(1020.0, "data_api")
        
        # Verify error message
        assert "No synchronization point established" in str(excinfo.value)
    
    def test_buffer_event(self, engine: SyncEngine, sample_match_event: MatchEvent) -> None:
        """Test buffering events."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        engine.event_buffer = mock_event_buffer
        
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2, 10.0)
        
        # Sync with kickoff
        engine.sync_with_kickoff("data_api", 1000.0, "hls", 1010.0)
        
        # Buffer event
        engine.buffer_event(sample_match_event)
        
        # Verify add_event was called with correct parameters
        mock_event_buffer.add_event.assert_called_once_with(
            sample_match_event, sample_match_event.timestamp, "data_api"
        )
        
        # Try to buffer with unregistered source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.buffer_event(sample_match_event, "unknown")
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        # Try to buffer without synchronization
        engine_unsync = SyncEngine()
        engine_unsync.register_timestamp_source("data_api", 1)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine_unsync.buffer_event(sample_match_event)
        
        # Verify error message
        assert "No synchronization point established" in str(excinfo.value)
    
    def test_get_events_for_audio(self, engine: SyncEngine, sample_match_event: MatchEvent) -> None:
        """Test getting events for audio timestamp."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        mock_event_buffer.get_events_in_window.return_value = [sample_match_event]
        engine.event_buffer = mock_event_buffer
        
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2, 10.0)
        
        # Sync with kickoff
        engine.sync_with_kickoff("data_api", 1000.0, "hls", 1010.0)
        
        # Get events for audio timestamp
        audio_timestamp = 1625097605.0 + 10.0
        events = engine.get_events_for_audio(audio_timestamp, "hls", 5.0)
        
        # Verify get_events_in_window was called with correct parameters
        aligned_timestamp = audio_timestamp - 10.0  # Adjusted for offset
        mock_event_buffer.get_events_in_window.assert_called_once_with(
            aligned_timestamp - 5.0, aligned_timestamp + 5.0
        )
        
        # Verify events
        assert len(events) == 1
        assert events[0] == sample_match_event
        
        # Try to get events with unregistered source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.get_events_for_audio(1625097605.0, "unknown")
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        # Try to get events without synchronization
        engine_unsync = SyncEngine()
        engine_unsync.register_timestamp_source("hls", 2)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine_unsync.get_events_for_audio(1625097605.0, "hls")
        
        # Verify error message
        assert "No synchronization point established" in str(excinfo.value)
    
    def test_get_events_by_type(self, engine: SyncEngine, sample_match_event: MatchEvent) -> None:
        """Test getting events by type."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        mock_event_buffer.get_events_by_type.return_value = [sample_match_event]
        engine.event_buffer = mock_event_buffer
        
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2, 10.0)
        
        # Sync with kickoff
        engine.sync_with_kickoff("data_api", 1000.0, "hls", 1010.0)
        
        # Get events by type without timestamp
        events = engine.get_events_by_type("goal")
        
        # Verify get_events_by_type was called with correct parameters
        mock_event_buffer.get_events_by_type.assert_called_once_with("goal")
        
        # Verify events
        assert len(events) == 1
        assert events[0] == sample_match_event
        
        # Reset mock
        mock_event_buffer.reset_mock()
        
        # Get events by type with timestamp
        audio_timestamp = 1625097605.0 + 10.0
        events = engine.get_events_by_type("goal", audio_timestamp, "hls", 5.0)
        
        # Verify get_events_by_type was called with correct parameters
        aligned_timestamp = audio_timestamp - 10.0  # Adjusted for offset
        mock_event_buffer.get_events_by_type.assert_called_once_with(
            "goal", (aligned_timestamp - 5.0, aligned_timestamp + 5.0)
        )
        
        # Try to get events with unregistered source
        with pytest.raises(SynchronizationError) as excinfo:
            engine.get_events_by_type("goal", 1625097605.0, "unknown")
        
        # Verify error message
        assert "not registered" in str(excinfo.value)
        
        # Try to get events without synchronization
        engine_unsync = SyncEngine()
        engine_unsync.register_timestamp_source("hls", 2)
        
        with pytest.raises(SynchronizationError) as excinfo:
            engine_unsync.get_events_by_type("goal", 1625097605.0, "hls")
        
        # Verify error message
        assert "No synchronization point established" in str(excinfo.value)
    
    def test_resolve_timestamp_conflict(self, engine: SyncEngine) -> None:
        """Test resolving timestamp conflicts."""
        # Register sources with different priorities
        engine.register_timestamp_source("data_api", 1)  # Highest priority
        engine.register_timestamp_source("hls", 2)
        engine.register_timestamp_source("backup", 3)
        
        # Resolve conflict
        timestamps = {
            "data_api": 1000.0,
            "hls": 1010.0,
            "backup": 1005.0
        }
        resolved = engine.resolve_timestamp_conflict(timestamps)
        
        # Verify highest priority source was selected
        assert resolved == 1000.0
        
        # Resolve conflict with subset of sources
        timestamps = {
            "hls": 1010.0,
            "backup": 1005.0
        }
        resolved = engine.resolve_timestamp_conflict(timestamps)
        
        # Verify highest priority source was selected
        assert resolved == 1010.0
        
        # Try to resolve with empty dict
        with pytest.raises(SynchronizationError) as excinfo:
            engine.resolve_timestamp_conflict({})
        
        # Verify error message
        assert "No timestamps provided" in str(excinfo.value)
        
        # Try to resolve with unregistered sources
        with pytest.raises(SynchronizationError) as excinfo:
            engine.resolve_timestamp_conflict({"unknown": 1000.0})
        
        # Verify error message
        assert "No valid timestamp sources provided" in str(excinfo.value)
    
    def test_clear_buffer(self, engine: SyncEngine) -> None:
        """Test clearing buffer."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        engine.event_buffer = mock_event_buffer
        
        # Clear buffer
        engine.clear_buffer()
        
        # Verify clear was called
        mock_event_buffer.clear.assert_called_once()
    
    def test_set_buffer_size(self, engine: SyncEngine) -> None:
        """Test setting buffer size."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        engine.event_buffer = mock_event_buffer
        
        # Set buffer size
        engine.set_buffer_size(120)
        
        # Verify set_buffer_size was called with correct parameter
        mock_event_buffer.set_buffer_size.assert_called_once_with(120)
    
    def test_get_buffer_stats(self, engine: SyncEngine) -> None:
        """Test getting buffer statistics."""
        # Mock event buffer
        mock_event_buffer = MagicMock()
        mock_event_buffer.get_stats.return_value = {
            "total_events": 3,
            "processed_events": 1,
            "unprocessed_events": 2,
            "event_types": {"goal": 2, "card": 1},
            "buffer_size": 60,
            "cleanup_interval": 10,
            "average_age": 15.0,
            "max_age": 30.0
        }
        engine.event_buffer = mock_event_buffer
        
        # Register sources
        engine.register_timestamp_source("data_api", 1)
        engine.register_timestamp_source("hls", 2)
        
        # Set kickoff timestamp
        engine.kickoff_timestamp = 1000.0
        
        # Get buffer stats
        stats = engine.get_buffer_stats()
        
        # Verify get_stats was called
        mock_event_buffer.get_stats.assert_called_once()
        
        # Verify stats
        assert stats["total_events"] == 3
        assert stats["processed_events"] == 1
        assert stats["unprocessed_events"] == 2
        assert stats["event_types"] == {"goal": 2, "card": 1}
        assert stats["buffer_size"] == 60
        assert stats["cleanup_interval"] == 10
        assert stats["average_age"] == 15.0
        assert stats["max_age"] == 30.0
        assert "synchronization" in stats
        assert stats["synchronization"]["kickoff_timestamp"] == 1000.0
        assert stats["synchronization"]["sources"] == 2
        assert set(stats["synchronization"]["source_names"]) == {"data_api", "hls"}
    
    @patch("crowd_sentiment_music_generator.services.synchronization.sync_engine.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, engine: SyncEngine) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(engine.register_timestamp_source, "__wrapped__")
        assert hasattr(engine.sync_with_kickoff, "__wrapped__")
        assert hasattr(engine.align_timestamp, "__wrapped__")
        assert hasattr(engine.buffer_event, "__wrapped__")
        assert hasattr(engine.get_events_for_audio, "__wrapped__")
        assert hasattr(engine.get_events_by_type, "__wrapped__")
        assert hasattr(engine.resolve_timestamp_conflict, "__wrapped__")
        assert hasattr(engine.clear_buffer, "__wrapped__")
        assert hasattr(engine.set_buffer_size, "__wrapped__")
        assert hasattr(engine.get_buffer_stats, "__wrapped__")