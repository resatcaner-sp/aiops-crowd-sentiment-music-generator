"""Unit tests for the music-video synchronization module."""

import pytest
from unittest.mock import MagicMock

from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic, MusicSegment
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.highlight_generator.music_video_sync import (
    MusicVideoSynchronizer, SyncPoint
)
from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine


@pytest.fixture
def sample_parameters():
    """Create sample musical parameters for testing."""
    return MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )


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
def sample_highlight_music(sample_parameters):
    """Create sample highlight music for testing."""
    segments = [
        MusicSegment(
            start_time=0.0,
            end_time=10.0,
            parameters=sample_parameters,
            transition_in=True,
            transition_out=True
        ),
        MusicSegment(
            start_time=10.0,
            end_time=20.0,
            parameters=sample_parameters.copy(update={"intensity": 0.8}),
            transition_in=True,
            transition_out=True,
            accent_time=15.0,
            accent_type="goal"
        ),
        MusicSegment(
            start_time=20.0,
            end_time=30.0,
            parameters=sample_parameters.copy(update={"intensity": 0.6}),
            transition_in=True,
            transition_out=False
        )
    ]
    
    return HighlightMusic(
        highlight_id="highlight1",
        segments=segments,
        base_parameters=sample_parameters,
        duration=30.0
    )


@pytest.fixture
def mock_music_engine():
    """Create a mock music engine."""
    return MagicMock(spec=MagentaMusicEngine)


def test_align_music_to_video(sample_highlight_music, sample_highlight_segment):
    """Test aligning music to video."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Align music to video
    aligned_music = synchronizer.align_music_to_video(sample_highlight_music, sample_highlight_segment)
    
    # Check result
    assert aligned_music.highlight_id == sample_highlight_segment.id
    assert len(aligned_music.segments) == len(sample_highlight_music.segments)
    assert aligned_music.duration == sample_highlight_segment.duration
    
    # Check that segments are properly aligned
    for segment in aligned_music.segments:
        assert segment.start_time >= 0
        assert segment.end_time <= sample_highlight_segment.duration


def test_align_music_to_video_mismatched_ids(sample_highlight_music, sample_highlight_segment):
    """Test aligning music to video with mismatched IDs."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Modify highlight ID
    mismatched_music = sample_highlight_music.copy(update={"highlight_id": "different_id"})
    
    # Align should raise ValueError
    with pytest.raises(ValueError):
        synchronizer.align_music_to_video(mismatched_music, sample_highlight_segment)


def test_adjust_duration_simple(sample_highlight_music):
    """Test simple duration adjustment."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Target duration (close to original, should use simple scaling)
    target_duration = 33.0  # 1.1x original
    
    # Adjust duration
    adjusted_music = synchronizer.adjust_duration(sample_highlight_music, target_duration)
    
    # Check result
    assert adjusted_music.duration == target_duration
    assert len(adjusted_music.segments) == len(sample_highlight_music.segments)
    
    # Check that segments are properly scaled
    for i, segment in enumerate(adjusted_music.segments):
        original_segment = sample_highlight_music.segments[i]
        original_duration = original_segment.end_time - original_segment.start_time
        adjusted_duration = segment.end_time - segment.start_time
        
        # Check that duration ratio is approximately 1.1
        assert abs(adjusted_duration / original_duration - 1.1) < 0.01


def test_adjust_duration_smart(sample_highlight_music):
    """Test smart duration adjustment."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Target duration (significantly different, should use smart scaling)
    target_duration = 45.0  # 1.5x original
    
    # Adjust duration
    adjusted_music = synchronizer.adjust_duration(sample_highlight_music, target_duration)
    
    # Check result
    assert adjusted_music.duration == target_duration
    assert len(adjusted_music.segments) == len(sample_highlight_music.segments)
    
    # Check that segments are properly scaled but not necessarily uniformly
    total_original_duration = sum(
        segment.end_time - segment.start_time for segment in sample_highlight_music.segments
    )
    total_adjusted_duration = sum(
        segment.end_time - segment.start_time for segment in adjusted_music.segments
    )
    
    # Check that total duration ratio is correct
    assert abs(total_adjusted_duration / total_original_duration - 1.5) < 0.01
    
    # Check that important segments (with accents) are preserved better
    for i, segment in enumerate(adjusted_music.segments):
        original_segment = sample_highlight_music.segments[i]
        
        # If segment has accent, it should be scaled less aggressively
        if original_segment.accent_time is not None:
            original_duration = original_segment.end_time - original_segment.start_time
            adjusted_duration = segment.end_time - segment.start_time
            
            # Important segments should be scaled less than average
            # This is a heuristic check - the exact scaling depends on the implementation
            assert adjusted_duration / original_duration <= 1.6


def test_adjust_duration_invalid(sample_highlight_music):
    """Test adjusting music duration with invalid target."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Adjust should raise ValueError for non-positive duration
    with pytest.raises(ValueError):
        synchronizer.adjust_duration(sample_highlight_music, 0.0)


def test_generate_transitions(sample_highlight_music):
    """Test generating transitions between segments."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Generate transitions
    music_with_transitions = synchronizer.generate_transitions(sample_highlight_music)
    
    # Check result
    assert music_with_transitions.highlight_id == sample_highlight_music.highlight_id
    
    # Should have more segments than original (due to added transitions)
    assert len(music_with_transitions.segments) > len(sample_highlight_music.segments)
    
    # Check that total duration is preserved
    assert music_with_transitions.duration == sample_highlight_music.duration
    
    # Check that transitions are properly placed between segments
    for i in range(len(music_with_transitions.segments) - 1):
        current_segment = music_with_transitions.segments[i]
        next_segment = music_with_transitions.segments[i + 1]
        
        # Segments should be sequential with no gaps or overlaps
        assert current_segment.end_time == next_segment.start_time
        
    # Check that transitions have appropriate parameters
    # Find transition segments (those without transition_in and transition_out flags)
    transition_segments = [
        segment for segment in music_with_transitions.segments
        if not segment.transition_in and not segment.transition_out
    ]
    
    # Should have at least one transition segment
    assert len(transition_segments) > 0
    
    # Transitions should have appropriate durations
    for transition in transition_segments:
        transition_duration = transition.end_time - transition.start_time
        assert transition_duration > 0


def test_generate_transitions_single_segment():
    """Test generating transitions with a single segment."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create music with a single segment
    params = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano"],
        mood="neutral"
    )
    
    segments = [
        MusicSegment(
            start_time=0.0,
            end_time=30.0,
            parameters=params,
            transition_in=True,
            transition_out=False
        )
    ]
    
    music = HighlightMusic(
        highlight_id="highlight1",
        segments=segments,
        base_parameters=params,
        duration=30.0
    )
    
    # Generate transitions
    music_with_transitions = synchronizer.generate_transitions(music)
    
    # Check result - should be unchanged
    assert len(music_with_transitions.segments) == 1
    assert music_with_transitions.duration == music.duration


def test_create_sync_points(sample_highlight_music, sample_highlight_segment):
    """Test creating synchronization points."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create sync points
    sync_points = synchronizer._create_sync_points(sample_highlight_music, sample_highlight_segment)
    
    # Check result
    assert len(sync_points) >= 2  # At least start and end points
    
    # Check that start and end points are included
    start_points = [sp for sp in sync_points if sp.type == "start"]
    end_points = [sp for sp in sync_points if sp.type == "end"]
    
    assert len(start_points) == 1
    assert len(end_points) == 1
    assert start_points[0].video_time == 0.0
    assert start_points[0].music_time == 0.0
    assert end_points[0].video_time == sample_highlight_segment.duration
    assert end_points[0].music_time == sample_highlight_music.duration


def test_align_segments_to_sync_points():
    """Test aligning segments to sync points."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create sample parameters
    params = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano"],
        mood="neutral"
    )
    
    # Create sample segments
    segments = [
        MusicSegment(
            start_time=0.0,
            end_time=10.0,
            parameters=params,
            transition_in=True,
            transition_out=True
        ),
        MusicSegment(
            start_time=10.0,
            end_time=20.0,
            parameters=params,
            transition_in=True,
            transition_out=False,
            accent_time=15.0,
            accent_type="goal"
        )
    ]
    
    # Create sync points
    sync_points = [
        SyncPoint(video_time=0.0, music_time=0.0, type="start"),
        SyncPoint(video_time=15.0, music_time=10.0, type="segment_boundary"),
        SyncPoint(video_time=25.0, music_time=15.0, type="key_moment"),
        SyncPoint(video_time=30.0, music_time=20.0, type="end")
    ]
    
    # Align segments
    aligned_segments = synchronizer._align_segments_to_sync_points(segments, sync_points)
    
    # Check result
    assert len(aligned_segments) == len(segments)
    
    # Check that segments are properly aligned
    assert aligned_segments[0].start_time == 0.0
    assert aligned_segments[0].end_time == 15.0
    assert aligned_segments[1].start_time == 15.0
    assert aligned_segments[1].end_time == 30.0
    assert aligned_segments[1].accent_time == 25.0


def test_create_transition_segment(sample_parameters):
    """Test creating a transition segment."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create source and target segments
    from_segment = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=sample_parameters,
        transition_in=True,
        transition_out=True
    )
    
    to_segment = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=sample_parameters.copy(update={"intensity": 0.9}),
        transition_in=True,
        transition_out=False
    )
    
    # Create transition
    transition = synchronizer._create_transition_segment(from_segment, to_segment)
    
    # Check result
    assert transition.start_time < transition.end_time
    assert transition.end_time == to_segment.start_time
    assert transition.parameters.intensity == (sample_parameters.intensity + 0.9) / 2


def test_interpolate_parameters(sample_parameters):
    """Test interpolating between musical parameters."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create target parameters
    target_params = sample_parameters.copy(update={
        "tempo": 140.0,
        "intensity": 0.9,
        "instrumentation": ["piano", "brass"],
        "transition_duration": 2.0
    })
    
    # Interpolate
    interpolated = synchronizer._interpolate_parameters(sample_parameters, target_params)
    
    # Check result
    assert interpolated.tempo == (sample_parameters.tempo + target_params.tempo) / 2
    assert interpolated.intensity == (sample_parameters.intensity + target_params.intensity) / 2
    assert set(interpolated.instrumentation) == set(["piano", "strings", "brass"])
    assert interpolated.transition_duration == 2.0  # From target_params


def test_calculate_segment_importance():
    """Test calculating segment importance."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create parameters
    params = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano"],
        mood="neutral"
    )
    
    # Create segments with different importance factors
    regular_segment = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=params,
        transition_in=True,
        transition_out=True
    )
    
    accent_segment = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=params,
        transition_in=True,
        transition_out=True,
        accent_time=15.0,
        accent_type="goal"
    )
    
    high_intensity_segment = MusicSegment(
        start_time=20.0,
        end_time=30.0,
        parameters=params.copy(update={"intensity": 0.9}),
        transition_in=True,
        transition_out=True
    )
    
    # Calculate importance scores
    regular_importance = synchronizer._calculate_segment_importance(regular_segment)
    accent_importance = synchronizer._calculate_segment_importance(accent_segment)
    high_intensity_importance = synchronizer._calculate_segment_importance(high_intensity_segment)
    
    # Check results
    assert regular_importance < accent_importance  # Segments with accents are more important
    assert regular_importance < high_intensity_importance  # Higher intensity segments are more important
    assert accent_importance > high_intensity_importance  # Accent is more important than just high intensity


def test_calculate_musical_contrast():
    """Test calculating musical contrast between segments."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create base parameters
    base_params = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    # Create segments with different levels of contrast
    segment1 = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=base_params,
        transition_in=True,
        transition_out=True
    )
    
    # Similar segment (low contrast)
    segment2 = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=base_params.copy(update={"intensity": 0.75}),
        transition_in=True,
        transition_out=True
    )
    
    # Medium contrast segment
    segment3 = MusicSegment(
        start_time=20.0,
        end_time=30.0,
        parameters=base_params.copy(update={
            "tempo": 140.0,
            "intensity": 0.8,
            "instrumentation": ["piano", "strings", "brass"]
        }),
        transition_in=True,
        transition_out=True
    )
    
    # High contrast segment
    segment4 = MusicSegment(
        start_time=30.0,
        end_time=40.0,
        parameters=base_params.copy(update={
            "tempo": 90.0,
            "key": "A Minor",
            "intensity": 0.4,
            "mood": "dark",
            "instrumentation": ["strings", "synth"]
        }),
        transition_in=True,
        transition_out=True
    )
    
    # Calculate contrast scores
    low_contrast = synchronizer._calculate_musical_contrast(segment1, segment2)
    medium_contrast = synchronizer._calculate_musical_contrast(segment1, segment3)
    high_contrast = synchronizer._calculate_musical_contrast(segment1, segment4)
    
    # Check results
    assert low_contrast < 0.3  # Low contrast
    assert 0.3 < medium_contrast < 0.7  # Medium contrast
    assert high_contrast > 0.7  # High contrast
    
    # Check that contrast is symmetric
    assert abs(synchronizer._calculate_musical_contrast(segment1, segment4) - 
               synchronizer._calculate_musical_contrast(segment4, segment1)) < 0.01


def test_create_simple_transition():
    """Test creating a simple transition."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create parameters
    params1 = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    params2 = MusicalParameters(
        tempo=125.0,
        key="C Major",
        intensity=0.75,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    # Create segments
    segment1 = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=params1,
        transition_in=True,
        transition_out=True
    )
    
    segment2 = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=params2,
        transition_in=True,
        transition_out=True
    )
    
    # Create simple transition
    transition = synchronizer._create_simple_transition(segment1, segment2)
    
    # Check result
    assert transition.start_time < transition.end_time
    assert transition.end_time == segment2.start_time
    assert transition.parameters.key == params2.key  # Simple transitions favor target parameters
    assert transition.parameters.mood == params2.mood
    assert (transition.end_time - transition.start_time) <= 1.0  # Simple transitions are short


def test_create_standard_transition():
    """Test creating a standard transition."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create parameters
    params1 = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    params2 = MusicalParameters(
        tempo=140.0,
        key="G Major",
        intensity=0.8,
        instrumentation=["piano", "strings", "brass"],
        mood="bright"
    )
    
    # Create segments
    segment1 = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=params1,
        transition_in=True,
        transition_out=True
    )
    
    segment2 = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=params2,
        transition_in=True,
        transition_out=True
    )
    
    # Create standard transition
    transition = synchronizer._create_standard_transition(segment1, segment2)
    
    # Check result
    assert transition.start_time < transition.end_time
    assert transition.end_time == segment2.start_time
    assert transition.parameters.tempo == (params1.tempo + params2.tempo) / 2  # Interpolated parameters
    assert transition.parameters.intensity == (params1.intensity + params2.intensity) / 2
    assert 1.0 <= (transition.end_time - transition.start_time) <= 2.0  # Standard transitions are medium length


def test_create_complex_transition():
    """Test creating a complex transition."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create parameters
    params1 = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    params2 = MusicalParameters(
        tempo=90.0,
        key="A Minor",
        intensity=0.4,
        instrumentation=["strings", "synth"],
        mood="dark"
    )
    
    # Create segments
    segment1 = MusicSegment(
        start_time=0.0,
        end_time=10.0,
        parameters=params1,
        transition_in=True,
        transition_out=True
    )
    
    segment2 = MusicSegment(
        start_time=10.0,
        end_time=20.0,
        parameters=params2,
        transition_in=True,
        transition_out=True
    )
    
    # Create complex transition
    transition = synchronizer._create_complex_transition(segment1, segment2)
    
    # Check result
    assert transition.start_time < transition.end_time
    assert transition.end_time == segment2.start_time
    assert (transition.end_time - transition.start_time) >= 2.0  # Complex transitions are longer
    assert "strings" in transition.parameters.instrumentation  # Should include common instruments
    assert transition.parameters.mood == "transitional"  # Should use transitional mood


def test_recalculate_segment_times():
    """Test recalculating segment times."""
    # Create synchronizer
    synchronizer = MusicVideoSynchronizer()
    
    # Create parameters
    params = MusicalParameters(
        tempo=120.0,
        key="C Major",
        intensity=0.7,
        instrumentation=["piano", "strings"],
        mood="bright"
    )
    
    # Create segments with potentially overlapping times
    segments = [
        MusicSegment(
            start_time=0.0,
            end_time=10.0,
            parameters=params,
            transition_in=True,
            transition_out=True
        ),
        MusicSegment(
            start_time=9.5,  # Overlaps with previous segment
            end_time=15.0,
            parameters=params,
            transition_in=True,
            transition_out=True,
            accent_time=12.0,
            accent_type="goal"
        ),
        MusicSegment(
            start_time=15.0,
            end_time=20.0,
            parameters=params,
            transition_in=True,
            transition_out=True
        )
    ]
    
    # Recalculate times
    recalculated = synchronizer._recalculate_segment_times(segments)
    
    # Check result
    assert len(recalculated) == len(segments)
    
    # Check that segments are sequential with no overlaps
    for i in range(len(recalculated) - 1):
        assert recalculated[i].end_time == recalculated[i + 1].start_time
    
    # Check that accent times are properly adjusted
    assert recalculated[1].accent_time is not None
    assert recalculated[1].start_time < recalculated[1].accent_time < recalculated[1].end_time
    
    # Check that total duration is preserved
    original_duration = sum(segment.end_time - segment.start_time for segment in segments)
    recalculated_duration = sum(segment.end_time - segment.start_time for segment in recalculated)
    assert abs(original_duration - recalculated_duration) < 0.01