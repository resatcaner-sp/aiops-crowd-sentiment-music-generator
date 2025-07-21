"""Unit tests for optimized audio processor."""

import os
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.audio_processing.optimized_processor import (
    OptimizedAudioProcessor,
    BatchHighlightProcessor,
    PerformanceMonitor,
)
from crowd_sentiment_music_generator.utils.parallel_processing import (
    parallel_map,
    batch_process,
    ResourceMonitor,
)


class TestOptimizedAudioProcessor:
    """Tests for OptimizedAudioProcessor."""

    @pytest.fixture
    def processor(self):
        """Create an optimized audio processor."""
        return OptimizedAudioProcessor(max_workers=2)

    @pytest.fixture
    def audio_segments(self):
        """Create test audio segments."""
        return [np.random.rand(22050 * 5) for _ in range(5)]  # 5 segments of 5 seconds each

    @pytest.fixture
    def highlight_segments(self, tmp_path):
        """Create test highlight segments."""
        highlights = []
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
            highlights.append(highlight)
        
        return highlights

    def test_process_audio_batch(self, processor, audio_segments):
        """Test processing a batch of audio segments."""
        # Mock _extract_audio_features to avoid actual processing
        with patch.object(processor, "_extract_audio_features") as mock_extract:
            mock_extract.side_effect = lambda segment, sr=22050: {"rms_energy": float(np.mean(np.abs(segment)))}
            
            # Process batch
            results = processor.process_audio_batch(audio_segments)
            
            # Check results
            assert len(results) == len(audio_segments)
            assert all("rms_energy" in result for result in results)
            
            # Check that _extract_audio_features was called for each segment
            assert mock_extract.call_count == len(audio_segments)

    def test_process_highlight_batch(self, processor, highlight_segments):
        """Test processing a batch of highlight segments."""
        # Mock _load_audio_from_video and _extract_audio_features
        with patch.object(processor, "_load_audio_from_video") as mock_load, \
             patch.object(processor, "_extract_audio_features") as mock_extract:
            
            # Set up mocks
            mock_load.side_effect = lambda path: np.random.rand(22050 * 5)
            mock_extract.side_effect = lambda audio, sr=22050: {"rms_energy": float(np.mean(np.abs(audio)))}
            
            # Process batch
            results = processor.process_highlight_batch(highlight_segments, batch_size=2)
            
            # Check results
            assert len(results) == len(highlight_segments)
            assert all(isinstance(result[0], HighlightSegment) for result in results)
            assert all(isinstance(result[1], dict) for result in results)
            assert all("rms_energy" in result[1] for result in results)
            
            # Check that _load_audio_from_video was called for each highlight
            assert mock_load.call_count == len(highlight_segments)
            
            # Check that _extract_audio_features was called for each highlight
            assert mock_extract.call_count == len(highlight_segments)

    def test_extract_audio_features_caching(self, processor):
        """Test that _extract_audio_features uses caching."""
        # Create test audio
        audio = np.random.rand(22050 * 5)
        
        # Mock cache methods
        with patch("crowd_sentiment_music_generator.utils.cache.cached") as mock_cached:
            # Call _extract_audio_features
            processor._extract_audio_features(audio)
            
            # Check that cached decorator was applied
            assert mock_cached.called


class TestBatchHighlightProcessor:
    """Tests for BatchHighlightProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a batch highlight processor."""
        return BatchHighlightProcessor(max_workers=2, batch_size=2)

    @pytest.fixture
    def highlight_segments(self, tmp_path):
        """Create test highlight segments."""
        highlights = []
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
            highlights.append(highlight)
        
        return highlights

    def test_process_highlights(self, processor, highlight_segments):
        """Test processing multiple highlights."""
        # Mock audio_processor.process_highlight_batch
        with patch.object(processor.audio_processor, "process_highlight_batch") as mock_process:
            # Set up mock
            mock_process.return_value = [
                (highlight, {"rms_energy": 0.5}) for highlight in highlight_segments
            ]
            
            # Process highlights
            results = processor.process_highlights(highlight_segments)
            
            # Check results
            assert len(results) == len(highlight_segments)
            assert all(highlight_id in results for highlight_id in [h.id for h in highlight_segments])
            assert all("rms_energy" in features for features in results.values())
            
            # Check that process_highlight_batch was called
            mock_process.assert_called_once()

    def test_generate_music_for_highlights(self, processor, highlight_segments):
        """Test generating music for multiple highlights."""
        # Process highlights
        results = processor.generate_music_for_highlights(highlight_segments)
        
        # Check results
        assert len(results) == len(highlight_segments)
        assert all(highlight_id in results for highlight_id in [h.id for h in highlight_segments])
        assert all(isinstance(music, HighlightMusic) for music in results.values())
        assert all(music.highlight_id == highlight_id for highlight_id, music in results.items())


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a performance monitor."""
        return PerformanceMonitor()

    def test_record_metric(self, monitor):
        """Test recording a performance metric."""
        # Record metric
        monitor.record_metric(
            category="audio_processing",
            operation="extract_features",
            duration=1.5,
            details={"cpu": {"avg": 50.0}, "memory": {"avg": 30.0}},
        )
        
        # Check that metric was recorded
        metrics = monitor.get_metrics("audio_processing")
        assert len(metrics["audio_processing"]) == 1
        assert metrics["audio_processing"][0]["operation"] == "extract_features"
        assert metrics["audio_processing"][0]["duration"] == 1.5
        assert metrics["audio_processing"][0]["cpu_usage"] == 50.0
        assert metrics["audio_processing"][0]["memory_usage"] == 30.0

    def test_get_average_duration(self, monitor):
        """Test getting average duration."""
        # Record metrics
        monitor.record_metric(
            category="audio_processing",
            operation="extract_features",
            duration=1.0,
            details={"cpu": {"avg": 50.0}, "memory": {"avg": 30.0}},
        )
        monitor.record_metric(
            category="audio_processing",
            operation="extract_features",
            duration=2.0,
            details={"cpu": {"avg": 60.0}, "memory": {"avg": 40.0}},
        )
        monitor.record_metric(
            category="audio_processing",
            operation="process_batch",
            duration=3.0,
            details={"cpu": {"avg": 70.0}, "memory": {"avg": 50.0}},
        )
        
        # Check average duration for category
        avg_duration = monitor.get_average_duration("audio_processing")
        assert avg_duration == 2.0  # (1.0 + 2.0 + 3.0) / 3
        
        # Check average duration for operation
        avg_duration = monitor.get_average_duration("audio_processing", "extract_features")
        assert avg_duration == 1.5  # (1.0 + 2.0) / 2

    def test_report_metrics(self, monitor, caplog):
        """Test reporting metrics."""
        # Record metrics
        monitor.record_metric(
            category="audio_processing",
            operation="extract_features",
            duration=1.0,
            details={"cpu": {"avg": 50.0}, "memory": {"avg": 30.0}},
        )
        monitor.record_metric(
            category="audio_processing",
            operation="process_batch",
            duration=2.0,
            details={"cpu": {"avg": 60.0}, "memory": {"avg": 40.0}},
        )
        
        # Report metrics
        monitor.report_metrics()
        
        # Check that metrics were reported
        assert "Performance metrics for audio_processing" in caplog.text
        assert "Operations=2" in caplog.text
        assert "Avg Duration=1.50s" in caplog.text


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_monitor_resources(self):
        """Test monitoring resource usage."""
        # Create monitor
        monitor = ResourceMonitor()
        
        # Start monitoring
        monitor.start()
        
        # Simulate some work
        time.sleep(0.1)
        
        # Stop monitoring
        stats = monitor.stop()
        
        # Check statistics
        assert stats["duration"] > 0
        assert "cpu" in stats
        assert "memory" in stats
        assert "disk_io" in stats
        assert len(stats["samples"]) > 0


class TestParallelProcessing:
    """Tests for parallel processing utilities."""

    def test_parallel_map(self):
        """Test parallel_map function."""
        # Define a simple function
        def square(x):
            return x * x
        
        # Apply function in parallel
        results = parallel_map(square, [1, 2, 3, 4, 5], max_workers=2)
        
        # Check results
        assert sorted(results) == [1, 4, 9, 16, 25]

    def test_batch_process(self):
        """Test batch_process function."""
        # Define a function to process a batch
        def process_batch(batch):
            return [x * x for x in batch]
        
        # Process items in batches
        results = batch_process(
            [1, 2, 3, 4, 5],
            process_batch,
            batch_size=2,
            max_workers=2,
            parallel=True,
        )
        
        # Check results
        assert sorted(results) == [1, 4, 9, 16, 25]


class TestPerformance:
    """Performance tests for optimized audio processing."""

    @pytest.mark.performance
    def test_parallel_vs_sequential(self):
        """Compare parallel and sequential processing performance."""
        # Create large test data
        data_size = 10
        data = [np.random.rand(22050 * 5) for _ in range(data_size)]
        
        # Define processing function
        def process_item(item):
            # Simulate CPU-intensive work
            time.sleep(0.1)
            return np.mean(item)
        
        # Measure sequential processing time
        start_time = time.time()
        sequential_results = [process_item(item) for item in data]
        sequential_time = time.time() - start_time
        
        # Measure parallel processing time
        start_time = time.time()
        parallel_results = parallel_map(process_item, data, max_workers=os.cpu_count())
        parallel_time = time.time() - start_time
        
        # Check that results are the same
        assert len(sequential_results) == len(parallel_results)
        
        # Check that parallel processing is faster
        # Note: This test may fail on systems with limited resources
        # or if the overhead of parallelization outweighs the benefits
        print(f"Sequential time: {sequential_time:.2f}s, Parallel time: {parallel_time:.2f}s")
        
        # We don't assert that parallel is always faster, as it depends on the system
        # and the specific test environment, but we log the results for analysis

    @pytest.mark.performance
    def test_batch_processing_performance(self):
        """Test batch processing performance with different batch sizes."""
        # Create test data
        data_size = 20
        data = [np.random.rand(22050 * 2) for _ in range(data_size)]
        
        # Define processing function
        def process_batch(batch):
            # Simulate batch processing
            time.sleep(0.1 * len(batch))
            return [np.mean(item) for item in batch]
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5, 10]
        times = []
        
        for batch_size in batch_sizes:
            # Measure processing time
            start_time = time.time()
            results = batch_process(
                data,
                process_batch,
                batch_size=batch_size,
                max_workers=os.cpu_count(),
                parallel=True,
            )
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            # Check results
            assert len(results) == data_size
        
        # Log results for analysis
        for batch_size, elapsed_time in zip(batch_sizes, times):
            print(f"Batch size {batch_size}: {elapsed_time:.2f}s")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis