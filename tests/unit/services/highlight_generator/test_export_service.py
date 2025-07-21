"""Unit tests for the export service."""

import os
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.export_error import ExportError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic, MusicSegment
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.highlight_generator.export_service import (
    ExportService, ExportFormat, QualityPreset, ExportOptions, ExportMetadata
)


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
def sample_highlight_segment(tmp_path):
    """Create a sample highlight segment for testing."""
    # Create a dummy video file
    video_path = tmp_path / "test_video.mp4"
    with open(video_path, "w") as f:
        f.write("Dummy video content")
    
    return HighlightSegment(
        id="highlight1",
        start_time=100.0,
        end_time=130.0,
        key_moment_time=115.0,
        video_path=str(video_path),
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
def export_service():
    """Create an export service for testing."""
    return ExportService()


def test_export_highlight_mp4(export_service, sample_highlight_segment, sample_highlight_music, tmp_path):
    """Test exporting a highlight as MP4."""
    # Create export options
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM,
        metadata=ExportMetadata(
            title="Test Highlight",
            description="A test highlight",
            tags=["test", "highlight"],
            author="Test Author"
        )
    )
    
    # Export highlight
    output_path = tmp_path / "output.mp4"
    result = export_service.export_highlight(
        sample_highlight_segment, sample_highlight_music, output_path, options
    )
    
    # Check result
    assert os.path.exists(result)
    assert result.endswith(".mp4")


def test_export_highlight_audio_only(export_service, sample_highlight_music, tmp_path):
    """Test exporting a highlight as audio only."""
    # Create export options
    options = ExportOptions(
        format=ExportFormat.MP3,
        quality_preset=QualityPreset.HIGH
    )
    
    # Export audio
    output_path = tmp_path / "output.mp3"
    result = export_service.export_audio_only(sample_highlight_music, output_path, options)
    
    # Check result
    assert os.path.exists(result)
    assert result.endswith(".mp3")


def test_export_highlight_mismatched_ids(export_service, sample_highlight_segment, sample_highlight_music, tmp_path):
    """Test exporting a highlight with mismatched IDs."""
    # Modify highlight ID
    mismatched_music = sample_highlight_music.copy(update={"highlight_id": "different_id"})
    
    # Create export options
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM
    )
    
    # Export should raise ValueError
    with pytest.raises(ValueError):
        export_service.export_highlight(
            sample_highlight_segment, mismatched_music, tmp_path / "output.mp4", options
        )


def test_export_highlight_missing_video(export_service, sample_highlight_segment, sample_highlight_music, tmp_path):
    """Test exporting a highlight with a missing video file."""
    # Modify video path to a non-existent file
    modified_segment = sample_highlight_segment.copy(update={"video_path": "/non/existent/path.mp4"})
    
    # Create export options
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM
    )
    
    # Export should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        export_service.export_highlight(
            modified_segment, sample_highlight_music, tmp_path / "output.mp4", options
        )


def test_export_audio_only_invalid_format(export_service, sample_highlight_music, tmp_path):
    """Test exporting audio with an invalid format."""
    # Create export options with a video format
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM
    )
    
    # Export should raise ValueError
    with pytest.raises(ValueError):
        export_service.export_audio_only(sample_highlight_music, tmp_path / "output.mp4", options)


def test_apply_quality_preset(export_service):
    """Test applying quality preset settings."""
    # Create options with minimal settings
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.HIGH
    )
    
    # Apply quality preset
    updated_options = export_service._apply_quality_preset(options)
    
    # Check that preset settings were applied
    assert updated_options.bitrate is not None
    assert updated_options.audio_bitrate is not None
    assert updated_options.resolution is not None
    assert updated_options.fps is not None
    
    # Check specific values from the HIGH preset for MP4
    assert updated_options.bitrate == 8000
    assert updated_options.audio_bitrate == 320
    assert updated_options.resolution == (1920, 1080)
    assert updated_options.fps == 30


def test_apply_quality_preset_custom(export_service):
    """Test applying custom quality preset."""
    # Create options with custom settings
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.CUSTOM,
        bitrate=10000,
        audio_bitrate=256,
        resolution=(1280, 720),
        fps=24
    )
    
    # Apply quality preset
    updated_options = export_service._apply_quality_preset(options)
    
    # Check that custom settings were preserved
    assert updated_options.bitrate == 10000
    assert updated_options.audio_bitrate == 256
    assert updated_options.resolution == (1280, 720)
    assert updated_options.fps == 24


def test_ensure_extension(export_service):
    """Test ensuring the correct file extension."""
    # Test with no extension
    assert export_service._ensure_extension("output", "mp4") == "output.mp4"
    
    # Test with different extension
    assert export_service._ensure_extension("output.wav", "mp4") == "output.mp4"
    
    # Test with correct extension
    assert export_service._ensure_extension("output.mp4", "mp4") == "output.mp4"
    
    # Test with path object
    from pathlib import Path
    assert export_service._ensure_extension(Path("output.wav"), "mp4") == "output.mp4"


def test_get_supported_formats(export_service):
    """Test getting supported formats."""
    formats = export_service.get_supported_formats()
    
    # Check result
    assert "video" in formats
    assert "audio" in formats
    assert "mp4" in formats["video"]
    assert "mp3" in formats["audio"]


def test_get_quality_presets(export_service):
    """Test getting quality presets."""
    presets = export_service.get_quality_presets()
    
    # Check result
    assert "medium" in presets
    assert "high" in presets
    assert "broadcast" in presets


@patch("os.makedirs")
def test_export_creates_directory(mock_makedirs, export_service, sample_highlight_segment, sample_highlight_music, tmp_path):
    """Test that export creates the output directory if it doesn't exist."""
    # Create export options
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM
    )
    
    # Create a path with a non-existent directory
    output_path = tmp_path / "subdir" / "output.mp4"
    
    # Export highlight
    export_service.export_highlight(sample_highlight_segment, sample_highlight_music, output_path, options)
    
    # Check that makedirs was called
    mock_makedirs.assert_called_once_with(os.path.dirname(output_path))


def test_export_with_metadata(export_service, sample_highlight_segment, sample_highlight_music, tmp_path):
    """Test exporting with metadata."""
    # Create export options with metadata
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.MEDIUM,
        metadata=ExportMetadata(
            title="Test Highlight",
            description="A test highlight",
            tags=["test", "highlight"],
            author="Test Author",
            copyright="© 2025 Test",
            creation_date="2025-07-21",
            custom_fields={"team1": "Home Team", "team2": "Away Team", "score": "2-1"}
        )
    )
    
    # Mock the _embed_metadata method to verify it's called
    with patch.object(export_service, "_embed_metadata") as mock_embed:
        # Export highlight
        output_path = tmp_path / "output.mp4"
        export_service.export_highlight(sample_highlight_segment, sample_highlight_music, output_path, options)
        
        # Check that _embed_metadata was called with the correct arguments
        mock_embed.assert_called_once()
        args = mock_embed.call_args[0]
        assert args[0].endswith(".mp4")
        assert args[1] == options.metadata
        assert args[2] == ExportFormat.MP4


def test_social_media_preset(export_service):
    """Test the social media quality preset."""
    # Create options with social media preset
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.SOCIAL
    )
    
    # Apply quality preset
    updated_options = export_service._apply_quality_preset(options)
    
    # Check that social media preset settings were applied
    assert updated_options.resolution == (1080, 1080)  # Square format for social media
    assert updated_options.bitrate == 5000
    assert updated_options.audio_bitrate == 256
    assert updated_options.fps == 30


def test_broadcast_preset(export_service):
    """Test the broadcast quality preset."""
    # Create options with broadcast preset
    options = ExportOptions(
        format=ExportFormat.MP4,
        quality_preset=QualityPreset.BROADCAST
    )
    
    # Apply quality preset
    updated_options = export_service._apply_quality_preset(options)
    
    # Check that broadcast preset settings were applied
    assert updated_options.resolution == (1920, 1080)
    assert updated_options.bitrate == 15000
    assert updated_options.audio_bitrate == 384
    assert updated_options.fps == 60  # Higher frame rate for broadcast