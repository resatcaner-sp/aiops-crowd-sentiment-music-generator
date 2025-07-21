"""Unit tests for the highlight processor module."""

import os
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.highlight_generator.highlight_processor import HighlightProcessor
from crowd_sentiment_music_generator.services.synchronization.sync_engine import SyncEngine


@pytest.fixture
def mock_sync_engine():
    """Create a mock synchronization engine."""
    mock_engine = MagicMock(spec=SyncEngine)
    mock_engine.align_timestamp.return_value = 100.0
    mock_engine.event_buffer.get_events_in_window.return_value = [
        MatchEvent(
            id="event1",
            type="goal",
            timestamp=100.0,
            team_id="team1"
        ),
        MatchEvent(
            id="event2",
            type="yellow_card",
            timestamp=105.0,
            team_id="team2"
        )
    ]
    return mock_engine


@pytest.fixture
def sample_highlight_segment():
    """Create a sample highlight segment for testing."""
    return HighlightSegment(
        id="highlight1",
        start_time=100.0,
        end_time=130.0,
        key_moment_time=115.0,
        video_path="test_video.mp4",
        events=[]
    )


@pytest.fixture
def sample_events():
    """Create sample match events for testing."""
    return [
        MatchEvent(
            id="event1",
            type="goal",
            timestamp=100.0,
            team_id="team1"
        ),
        MatchEvent(
            id="event2",
            type="yellow_card",
            timestamp=105.0,
            team_id="team2"
        ),
        MatchEvent(
            id="event3",
            type="shot_on_target",
            timestamp=110.0,
            team_id="team1"
        )
    ]


def test_process_highlight(sample_highlight_segment):
    """Test processing a highlight segment."""
    # Create processor
    processor = HighlightProcessor()
    
    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        # Process highlight
        result = processor.process_highlight(sample_highlight_segment)
        
        # Check result
        assert result.id == sample_highlight_segment.id
        assert result.metadata is not None
        assert result.metadata["processed"] is True
        assert result.metadata["duration"] == 30.0  # end_time - start_time
        assert "frame_rate" in result.metadata
        assert "resolution" in result.metadata


def test_process_highlight_invalid_timestamps():
    """Test processing a highlight segment with invalid timestamps."""
    # Create processor
    processor = HighlightProcessor()
    
    # Create invalid segment (end_time <= start_time)
    invalid_segment = HighlightSegment(
        id="invalid",
        start_time=100.0,
        end_time=100.0,  # Equal to start_time
        key_moment_time=100.0,
        video_path="test_video.mp4",
        events=[]
    )
    
    # Process should raise ValueError
    with pytest.raises(ValueError):
        processor.process_highlight(invalid_segment)


def test_process_highlight_file_not_found(sample_highlight_segment):
    """Test processing a highlight segment with a non-existent video file."""
    # Create processor
    processor = HighlightProcessor()
    
    # Mock os.path.exists to return False
    with patch("os.path.exists", return_value=False):
        # Process should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            processor.process_highlight(sample_highlight_segment)


def test_extract_events_no_sync_engine(sample_highlight_segment):
    """Test extracting events without a sync engine."""
    # Create processor without sync engine
    processor = HighlightProcessor()
    
    # Extract should raise SynchronizationError
    with pytest.raises(SynchronizationError):
        processor.extract_events(sample_highlight_segment)


def test_extract_events(mock_sync_engine, sample_highlight_segment):
    """Test extracting events from a highlight segment."""
    # Create processor with mock sync engine
    processor = HighlightProcessor(sync_engine=mock_sync_engine)
    
    # Extract events
    events = processor.extract_events(sample_highlight_segment)
    
    # Check result
    assert len(events) == 2
    assert events[0].id == "event1"
    assert events[0].type == "goal"
    assert events[1].id == "event2"
    assert events[1].type == "yellow_card"
    
    # Verify sync engine methods were called
    mock_sync_engine.align_timestamp.assert_called()
    mock_sync_engine.event_buffer.get_events_in_window.assert_called_once()


def test_generate_music_composition_no_events(sample_highlight_segment):
    """Test generating a music composition with no events."""
    # Create processor
    processor = HighlightProcessor()
    
    # Generate composition
    composition = processor.generate_music_composition(sample_highlight_segment, [])
    
    # Check result
    assert isinstance(composition, HighlightMusic)
    assert composition.highlight_id == sample_highlight_segment.id
    assert len(composition.segments) == 1
    assert composition.segments[0].start_time == 0.0
    assert composition.segments[0].end_time == 30.0  # segment duration


def test_generate_music_composition_with_events(sample_highlight_segment, sample_events):
    """Test generating a music composition with events."""
    # Create processor
    processor = HighlightProcessor()
    
    # Adjust event timestamps to be within segment
    for i, event in enumerate(sample_events):
        event.timestamp = sample_highlight_segment.start_time + (i * 5.0)
    
    # Generate composition
    composition = processor.generate_music_composition(sample_highlight_segment, sample_events)
    
    # Check result
    assert isinstance(composition, HighlightMusic)
    assert composition.highlight_id == sample_highlight_segment.id
    assert len(composition.segments) > 0
    
    # Check that segments cover the entire duration
    covered_time = sum(seg.end_time - seg.start_time for seg in composition.segments)
    assert abs(covered_time - sample_highlight_segment.duration) < 0.1
    
    # Check that segments have appropriate parameters
    for segment in composition.segments:
        assert segment.parameters.tempo > 0
        assert segment.parameters.intensity >= 0 and segment.parameters.intensity <= 1
        assert segment.parameters.key is not None
        assert len(segment.parameters.instrumentation) > 0


def test_determine_base_parameters(sample_events):
    """Test determining base parameters from events."""
    # Create processor
    processor = HighlightProcessor()
    
    # Get base parameters
    params = processor._determine_base_parameters(sample_events)
    
    # Check result
    assert params.tempo >= 80.0 and params.tempo <= 140.0
    assert params.intensity >= 0.0 and params.intensity <= 1.0
    assert params.key is not None
    assert len(params.instrumentation) > 0
    assert params.mood in ["bright", "dark", "tense", "neutral"]


def test_create_music_segments(sample_highlight_segment, sample_events):
    """Test creating music segments from events."""
    # Create processor
    processor = HighlightProcessor()
    
    # Adjust event timestamps to be within segment
    for i, event in enumerate(sample_events):
        event.timestamp = sample_highlight_segment.start_time + (i * 10.0)
    
    # Get base parameters
    base_params = processor._determine_base_parameters(sample_events)
    
    # Create segments
    segments = processor._create_music_segments(sample_highlight_segment, sample_events, base_params)
    
    # Check result
    assert len(segments) > 0
    
    # Check that segments are properly ordered
    for i in range(len(segments) - 1):
        assert segments[i].end_time == segments[i + 1].start_time
    
    # Check that first segment starts at 0 and last segment ends at duration
    assert segments[0].start_time == 0.0
    assert segments[-1].end_time == sample_highlight_segment.duration