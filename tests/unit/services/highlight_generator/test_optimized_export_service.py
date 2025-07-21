"""Unit tests for optimized export service."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.highlight_generator.export_service import (
    ExportFormat,
    QualityPreset,
    ExportOptions,
    ExportMetadata,
)
from crowd_sentiment_music_generator.services.highlight_generator.optimized_export_service import (
    OptimizedExportService,
    BatchExportRequest,
    BatchExportResult,
)


class TestOptimizedExportService:
    """Tests for OptimizedExportService."""

    @pytest.fixture
    def export_service(self):
        """Create an optimized export service."""
        return OptimizedExportService(max_workers=2, batch_size=2)

    @pytest.fixture
    def highlight_segments(self, tmp_path):
        """Create test highlight segments."""
        highlights = {}
        for i in range(5):
            # Create dummy video file
            video_path = tmp_path / f"test_video_{i}.mp4"
            with open(video_path, "w") as f:
                f.write(f"Dummy video content {i}")
            
            # Create highlight segment
            highlight = HighlightSegment(
                id=f"highlight_{i}",
                match_id="test_match",
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                key_moment=i * 30.0 + 15.0,
                video_path=str(video_path),
            )
            highlights[f"highlight_{i}"] = highlight
        
        return highlights

    @pytest.fixture
    def highlight_music(self):
        """Create test highlight music."""
        music = {}
        for i in range(5):
            # Create highlight music
            music[f"highlight_{i}"] = HighlightMusic(
                highlight_id=f"highlight_{i}",
                duration=30.0,
                audio_data=b"test_audio_data",
            )
        
        return music

    @pytest.fixture
    def export_requests(self, tmp_path):
        """Create test export requests."""
        requests = []
        for i in range(5):
            # Create export request
            request = BatchExportRequest(
                highlight_id=f"highlight_{i}",
                output_path=str(tmp_path / f"output_{i}.mp4"),
                options=ExportOptions(
                    format=ExportFormat.MP4,
                    quality_preset=QualityPreset.MEDIUM,
                ),
            )
            requests.append(request)
        
        return requests

    def test_export_highlight_batch(self, export_service, highlight_segments, highlight_music, export_requests):
        """Test exporting a batch of highlights."""
        # Mock export_highlight to avoid actual export
        with patch.object(export_service, "export_highlight") as mock_export:
            # Set up mock
            mock_export.side_effect = lambda hs, hm, op, opts: op
            
            # Export batch
            results = export_service.export_highlight_batch(
                highlight_segments, highlight_music, export_requests
            )
            
            # Check results
            assert len(results) == len(export_requests)
            assert all(isinstance(result, BatchExportResult) for result in results)
            assert all(result.success for result in results)
            assert all(result.highlight_id == f"highlight_{i}" for i, result in enumerate(results))
            
            # Check that export_highlight was called for each request
            assert mock_export.call_count == len(export_requests)

    def test_export_highlight_batch_missing_highlight(self, export_service, highlight_segments, highlight_music, export_requests):
        """Test exporting a batch with a missing highlight."""
        # Remove one highlight
        del highlight_segments["highlight_2"]
        
        # Mock export_highlight to avoid actual export
        with patch.object(export_service, "export_highlight") as mock_export:
            # Set up mock
            mock_export.side_effect = lambda hs, hm, op, opts: op
            
            # Export batch
            results = export_service.export_highlight_batch(
                highlight_segments, highlight_music, export_requests
            )
            
            # Check results
            assert len(results) == len(export_requests)
            
            # Check that the missing highlight has an error
            for result in results:
                if result.highlight_id == "highlight_2":
                    assert not result.success
                    assert "not found" in result.error
                else:
                    assert result.success
            
            # Check that export_highlight was called for each valid request
            assert mock_export.call_count == len(export_requests) - 1

    def test_export_audio_batch(self, export_service, highlight_music, export_requests):
        """Test exporting a batch of audio files."""
        # Modify requests to use audio format
        for request in export_requests:
            request.options.format = ExportFormat.MP3
        
        # Mock export_audio_only to avoid actual export
        with patch.object(export_service, "export_audio_only") as mock_export:
            # Set up mock
            mock_export.side_effect = lambda hm, op, opts: op
            
            # Export batch
            results = export_service.export_audio_batch(highlight_music, export_requests)
            
            # Check results
            assert len(results) == len(export_requests)
            assert all(isinstance(result, BatchExportResult) for result in results)
            assert all(result.success for result in results)
            assert all(result.highlight_id == f"highlight_{i}" for i, result in enumerate(results))
            
            # Check that export_audio_only was called for each request
            assert mock_export.call_count == len(export_requests)

    def test_export_audio_batch_missing_music(self, export_service, highlight_music, export_requests):
        """Test exporting a batch with missing music."""
        # Remove one music
        del highlight_music["highlight_2"]
        
        # Modify requests to use audio format
        for request in export_requests:
            request.options.format = ExportFormat.MP3
        
        # Mock export_audio_only to avoid actual export
        with patch.object(export_service, "export_audio_only") as mock_export:
            # Set up mock
            mock_export.side_effect = lambda hm, op, opts: op
            
            # Export batch
            results = export_service.export_audio_batch(highlight_music, export_requests)
            
            # Check results
            assert len(results) == len(export_requests)
            
            # Check that the missing music has an error
            for result in results:
                if result.highlight_id == "highlight_2":
                    assert not result.success
                    assert "not found" in result.error
                else:
                    assert result.success
            
            # Check that export_audio_only was called for each valid request
            assert mock_export.call_count == len(export_requests) - 1

    def test_get_resource_usage(self, export_service):
        """Test getting resource usage statistics."""
        # Get resource usage
        usage = export_service.get_resource_usage()
        
        # Check result
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "disk_io" in usage

    def test_optimize_batch_size(self, export_service):
        """Test optimizing batch size."""
        # Optimize batch size
        batch_size = export_service.optimize_batch_size()
        
        # Check result
        assert batch_size > 0
        assert isinstance(batch_size, int)


@pytest.mark.performance
class TestExportServicePerformance:
    """Performance tests for optimized export service."""

    @pytest.fixture
    def export_service(self):
        """Create an optimized export service."""
        return OptimizedExportService(max_workers=os.cpu_count(), batch_size=2)

    @pytest.fixture
    def highlight_segments(self, tmp_path):
        """Create test highlight segments."""
        highlights = {}
        for i in range(10):
            # Create dummy video file
            video_path = tmp_path / f"test_video_{i}.mp4"
            with open(video_path, "w") as f:
                f.write(f"Dummy video content {i}")
            
            # Create highlight segment
            highlight = HighlightSegment(
                id=f"highlight_{i}",
                match_id="test_match",
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                key_moment=i * 30.0 + 15.0,
                video_path=str(video_path),
            )
            highlights[f"highlight_{i}"] = highlight
        
        return highlights

    @pytest.fixture
    def highlight_music(self):
        """Create test highlight music."""
        music = {}
        for i in range(10):
            # Create highlight music
            music[f"highlight_{i}"] = HighlightMusic(
                highlight_id=f"highlight_{i}",
                duration=30.0,
                audio_data=b"test_audio_data",
            )
        
        return music

    @pytest.fixture
    def export_requests(self, tmp_path):
        """Create test export requests."""
        requests = []
        for i in range(10):
            # Create export request
            request = BatchExportRequest(
                highlight_id=f"highlight_{i}",
                output_path=str(tmp_path / f"output_{i}.mp4"),
                options=ExportOptions(
                    format=ExportFormat.MP4,
                    quality_preset=QualityPreset.MEDIUM,
                ),
            )
            requests.append(request)
        
        return requests

    def test_parallel_vs_sequential_export(self, export_service, highlight_segments, highlight_music, export_requests, tmp_path):
        """Compare parallel and sequential export performance."""
        # Create a sequential export service
        sequential_service = OptimizedExportService(max_workers=1, batch_size=1)
        
        # Mock export_highlight to simulate export with delay
        def mock_export(hs, hm, op, opts):
            # Simulate export with delay
            time.sleep(0.1)
            return op
        
        # Test sequential export
        with patch.object(sequential_service, "export_highlight", side_effect=mock_export):
            start_time = time.time()
            sequential_results = sequential_service.export_highlight_batch(
                highlight_segments, highlight_music, export_requests
            )
            sequential_time = time.time() - start_time
        
        # Test parallel export
        with patch.object(export_service, "export_highlight", side_effect=mock_export):
            start_time = time.time()
            parallel_results = export_service.export_highlight_batch(
                highlight_segments, highlight_music, export_requests
            )
            parallel_time = time.time() - start_time
        
        # Check that results are the same
        assert len(sequential_results) == len(parallel_results)
        
        # Log performance results
        print(f"Sequential export time: {sequential_time:.2f}s")
        print(f"Parallel export time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis

    def test_batch_size_impact(self, export_service, highlight_segments, highlight_music, export_requests):
        """Test the impact of batch size on export performance."""
        # Mock export_highlight to simulate export with delay
        def mock_export(hs, hm, op, opts):
            # Simulate export with delay
            time.sleep(0.1)
            return op
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5, 10]
        times = []
        
        for batch_size in batch_sizes:
            # Create service with specific batch size
            service = OptimizedExportService(max_workers=os.cpu_count(), batch_size=batch_size)
            
            # Measure export time
            with patch.object(service, "export_highlight", side_effect=mock_export):
                start_time = time.time()
                results = service.export_highlight_batch(
                    highlight_segments, highlight_music, export_requests
                )
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
            
            # Check results
            assert len(results) == len(export_requests)
        
        # Log results for analysis
        for batch_size, elapsed_time in zip(batch_sizes, times):
            print(f"Batch size {batch_size}: {elapsed_time:.2f}s")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis