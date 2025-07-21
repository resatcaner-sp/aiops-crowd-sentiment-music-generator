"""Unit tests for export service caching functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.config.cache_config import CacheSettings
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.highlight_generator.export_service import (
    ExportService,
    ExportFormat,
    QualityPreset,
    ExportOptions,
)
from crowd_sentiment_music_generator.utils.cache import CacheManager


class TestExportServiceCache:
    """Tests for export service caching functionality."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        mock = MagicMock(spec=CacheManager)
        mock.cache = MagicMock()
        mock.cache.get_json.return_value = None
        mock.set_with_tags.return_value = True
        mock.invalidate_by_tag.return_value = 2
        return mock

    @pytest.fixture
    def export_service(self, mock_cache_manager):
        """Create an export service with a mock cache manager."""
        service = ExportService()
        service.cache_manager = mock_cache_manager
        service.cache_settings = CacheSettings(enabled=True)
        return service

    @pytest.fixture
    def highlight_segment(self):
        """Create a test highlight segment."""
        return HighlightSegment(
            id="test_highlight_1",
            match_id="test_match_1",
            start_time=60.0,
            end_time=90.0,
            key_moment=75.0,
            video_path="/tmp/test_video.mp4",
        )

    @pytest.fixture
    def highlight_music(self):
        """Create test highlight music."""
        return HighlightMusic(
            highlight_id="test_highlight_1",
            duration=30.0,
            audio_data=b"test_audio_data",
        )

    @pytest.fixture
    def export_options(self):
        """Create test export options."""
        return ExportOptions(
            format=ExportFormat.MP4,
            quality_preset=QualityPreset.MEDIUM,
        )

    def test_cache_highlight_export_result(self, export_service, mock_cache_manager):
        """Test caching highlight export result."""
        # Mock os.path.exists and os.path.getmtime
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=12345.0
        ):
            result = export_service.cache_highlight_export_result(
                "test_highlight_1",
                ExportFormat.MP4,
                QualityPreset.MEDIUM,
                "/tmp/output.mp4",
            )
            
            assert result is True
            mock_cache_manager.set_with_tags.assert_called_once()
            
            # Check the cache key and tags
            args = mock_cache_manager.set_with_tags.call_args
            assert args[0][0] == "highlight_export:test_highlight_1:mp4:medium"
            assert "highlight:test_highlight_1" in args[0][2]
            assert "exports" in args[0][2]
            assert "format:mp4" in args[0][2]

    def test_get_cached_export_hit(self, export_service, mock_cache_manager):
        """Test getting cached export with a cache hit."""
        # Set up cache hit
        mock_cache_manager.cache.get_json.return_value = {
            "highlight_id": "test_highlight_1",
            "format": "mp4",
            "quality": "medium",
            "output_path": "/tmp/output.mp4",
            "timestamp": 12345.0,
        }
        
        # Mock os.path.exists and os.path.getmtime
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=12345.0
        ):
            result = export_service.get_cached_export(
                "test_highlight_1", ExportFormat.MP4, QualityPreset.MEDIUM
            )
            
            assert result is not None
            assert result["highlight_id"] == "test_highlight_1"
            assert result["format"] == "mp4"
            assert result["output_path"] == "/tmp/output.mp4"
            mock_cache_manager.cache.get_json.assert_called_with(
                "highlight_export:test_highlight_1:mp4:medium"
            )

    def test_get_cached_export_miss(self, export_service, mock_cache_manager):
        """Test getting cached export with a cache miss."""
        # Set up cache miss
        mock_cache_manager.cache.get_json.return_value = None
        
        result = export_service.get_cached_export(
            "test_highlight_1", ExportFormat.MP4, QualityPreset.MEDIUM
        )
        
        assert result is None
        mock_cache_manager.cache.get_json.assert_called_with(
            "highlight_export:test_highlight_1:mp4:medium"
        )

    def test_get_cached_export_file_missing(self, export_service, mock_cache_manager):
        """Test getting cached export when the file is missing."""
        # Set up cache hit but file missing
        mock_cache_manager.cache.get_json.return_value = {
            "highlight_id": "test_highlight_1",
            "format": "mp4",
            "quality": "medium",
            "output_path": "/tmp/output.mp4",
            "timestamp": 12345.0,
        }
        
        # Mock os.path.exists to return False
        with patch("os.path.exists", return_value=False):
            result = export_service.get_cached_export(
                "test_highlight_1", ExportFormat.MP4, QualityPreset.MEDIUM
            )
            
            assert result is None

    def test_get_cached_export_file_modified(self, export_service, mock_cache_manager):
        """Test getting cached export when the file has been modified."""
        # Set up cache hit but file modified
        mock_cache_manager.cache.get_json.return_value = {
            "highlight_id": "test_highlight_1",
            "format": "mp4",
            "quality": "medium",
            "output_path": "/tmp/output.mp4",
            "timestamp": 12345.0,
        }
        
        # Mock os.path.exists and os.path.getmtime with different timestamp
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=54321.0
        ):
            result = export_service.get_cached_export(
                "test_highlight_1", ExportFormat.MP4, QualityPreset.MEDIUM
            )
            
            assert result is None

    def test_invalidate_highlight_cache(self, export_service, mock_cache_manager):
        """Test invalidating highlight cache."""
        result = export_service.invalidate_highlight_cache("test_highlight_1")
        
        assert result == 2
        mock_cache_manager.invalidate_by_tag.assert_called_with("highlight:test_highlight_1")

    def test_export_highlight_with_cache_hit(
        self, export_service, mock_cache_manager, highlight_segment, highlight_music, export_options
    ):
        """Test export_highlight with a cache hit."""
        # Set up cache hit
        mock_cache_manager.cache.get_json.return_value = {
            "highlight_id": "test_highlight_1",
            "format": "mp4",
            "quality": "medium",
            "output_path": "/tmp/output.mp4",
            "timestamp": 12345.0,
        }
        
        # Mock file operations
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=12345.0
        ), patch.object(
            export_service, "_export_video_with_audio", return_value="/tmp/output.mp4"
        ):
            result = export_service.export_highlight(
                highlight_segment, highlight_music, "/tmp/output.mp4", export_options
            )
            
            assert result == "/tmp/output.mp4"
            # Verify that _export_video_with_audio was not called (used cache)
            export_service._export_video_with_audio.assert_not_called()

    def test_export_highlight_with_cache_miss(
        self, export_service, mock_cache_manager, highlight_segment, highlight_music, export_options
    ):
        """Test export_highlight with a cache miss."""
        # Set up cache miss
        mock_cache_manager.cache.get_json.return_value = None
        
        # Mock file operations
        with patch("os.path.exists", return_value=True), patch(
            "os.makedirs", return_value=None
        ), patch.object(
            export_service, "_export_video_with_audio", return_value="/tmp/output.mp4"
        ), patch.object(
            export_service, "_embed_metadata", return_value=None
        ):
            result = export_service.export_highlight(
                highlight_segment, highlight_music, "/tmp/output.mp4", export_options
            )
            
            assert result == "/tmp/output.mp4"
            # Verify that _export_video_with_audio was called (cache miss)
            export_service._export_video_with_audio.assert_called_once()
            # Verify that the result was cached
            mock_cache_manager.set_with_tags.assert_called_once()

    def test_export_audio_only_with_cache_hit(
        self, export_service, mock_cache_manager, highlight_music
    ):
        """Test export_audio_only with a cache hit."""
        # Set up cache hit
        mock_cache_manager.cache.get_json.return_value = {
            "highlight_id": "test_highlight_1",
            "format": "mp3",
            "quality": "high",
            "output_path": "/tmp/output.mp3",
            "timestamp": 12345.0,
        }
        
        # Create audio export options
        audio_options = ExportOptions(
            format=ExportFormat.MP3,
            quality_preset=QualityPreset.HIGH,
        )
        
        # Mock file operations
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=12345.0
        ), patch.object(
            export_service, "_export_audio_only", return_value="/tmp/output.mp3"
        ):
            result = export_service.export_audio_only(
                highlight_music, "/tmp/output.mp3", audio_options
            )
            
            assert result == "/tmp/output.mp3"
            # Verify that _export_audio_only was not called (used cache)
            export_service._export_audio_only.assert_not_called()

    def test_export_audio_only_with_cache_miss(
        self, export_service, mock_cache_manager, highlight_music
    ):
        """Test export_audio_only with a cache miss."""
        # Set up cache miss
        mock_cache_manager.cache.get_json.return_value = None
        
        # Create audio export options
        audio_options = ExportOptions(
            format=ExportFormat.MP3,
            quality_preset=QualityPreset.HIGH,
        )
        
        # Mock file operations
        with patch("os.path.exists", return_value=True), patch(
            "os.makedirs", return_value=None
        ), patch.object(
            export_service, "_export_audio_only", return_value="/tmp/output.mp3"
        ), patch.object(
            export_service, "_embed_metadata", return_value=None
        ):
            result = export_service.export_audio_only(
                highlight_music, "/tmp/output.mp3", audio_options
            )
            
            assert result == "/tmp/output.mp3"
            # Verify that _export_audio_only was called (cache miss)
            export_service._export_audio_only.assert_called_once()
            # Verify that the result was cached
            mock_cache_manager.set_with_tags.assert_called_once()

    def test_cached_get_supported_formats(self, export_service):
        """Test that get_supported_formats is cached."""
        with patch("crowd_sentiment_music_generator.utils.cache.cached") as mock_cached:
            # Call the method to verify it's decorated
            export_service.get_supported_formats()
            
            # Check that the cached decorator was applied
            assert mock_cached.called

    def test_cached_get_quality_presets(self, export_service):
        """Test that get_quality_presets is cached."""
        with patch("crowd_sentiment_music_generator.utils.cache.cached") as mock_cached:
            # Call the method to verify it's decorated
            export_service.get_quality_presets()
            
            # Check that the cached decorator was applied
            assert mock_cached.called