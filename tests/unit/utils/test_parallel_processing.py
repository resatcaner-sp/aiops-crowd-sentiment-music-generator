"""Unit tests for parallel processing utilities."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.utils.parallel_processing import (
    ResourceMonitor,
    parallel_map,
    batch_process,
    monitor_resources,
)


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a resource monitor."""
        return ResourceMonitor(sampling_interval=0.1)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_io_counters")
    def test_start_stop(self, mock_disk_io, mock_memory, mock_cpu, monitor):
        """Test starting and stopping resource monitoring."""
        # Set up mocks
        mock_cpu.return_value = 10.0
        mock_memory.return_value.percent = 30.0
        mock_disk_io.return_value._asdict.return_value = {
            "read_bytes": 1000,
            "write_bytes": 2000,
            "read_count": 10,
            "write_count": 20,
        }
        
        # Start monitoring
        monitor.start()
        
        # Wait for a sample
        time.sleep(0.2)
        
        # Stop monitoring
        stats = monitor.stop()
        
        # Check results
        assert stats["duration"] > 0
        assert stats["cpu"]["start"] == 10.0
        assert stats["memory"]["start"] == 30.0
        assert len(stats["samples"]) > 0

    def test_sample_resources(self, monitor):
        """Test sampling resources."""
        # Mock _sample_resources to avoid actual sampling
        with patch.object(monitor, "_sample_resources"):
            # Start and stop monitoring
            monitor.start()
            stats = monitor.stop()
            
            # Check results
            assert stats["duration"] > 0


class TestParallelMap:
    """Tests for parallel_map function."""

    def test_parallel_map_empty(self):
        """Test parallel_map with empty list."""
        result = parallel_map(lambda x: x * 2, [])
        assert result == []

    def test_parallel_map_with_processes(self):
        """Test parallel_map with processes."""
        # Define a simple function
        def square(x):
            return x * x
        
        # Apply function in parallel
        result = parallel_map(square, [1, 2, 3, 4, 5], max_workers=2, use_processes=True)
        
        # Check result
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_parallel_map_with_threads(self):
        """Test parallel_map with threads."""
        # Define a simple function
        def square(x):
            return x * x
        
        # Apply function in parallel
        result = parallel_map(square, [1, 2, 3, 4, 5], max_workers=2, use_processes=False)
        
        # Check result
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_parallel_map_with_error(self):
        """Test parallel_map with function that raises an error."""
        # Define a function that raises an error for some inputs
        def risky_function(x):
            if x == 3:
                raise ValueError("Error for input 3")
            return x * x
        
        # Apply function in parallel
        with pytest.raises(ValueError):
            parallel_map(risky_function, [1, 2, 3, 4, 5], max_workers=2)


class TestBatchProcess:
    """Tests for batch_process function."""

    def test_batch_process_empty(self):
        """Test batch_process with empty list."""
        result = batch_process([], lambda batch: [x * 2 for x in batch])
        assert result == []

    def test_batch_process_sequential(self):
        """Test batch_process in sequential mode."""
        # Define a function to process a batch
        def process_batch(batch):
            return [x * x for x in batch]
        
        # Process items in batches
        result = batch_process(
            [1, 2, 3, 4, 5],
            process_batch,
            batch_size=2,
            parallel=False,
        )
        
        # Check result
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_batch_process_parallel(self):
        """Test batch_process in parallel mode."""
        # Define a function to process a batch
        def process_batch(batch):
            return [x * x for x in batch]
        
        # Process items in batches
        result = batch_process(
            [1, 2, 3, 4, 5],
            process_batch,
            batch_size=2,
            max_workers=2,
            parallel=True,
        )
        
        # Check result
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_batch_process_single_batch(self):
        """Test batch_process with a single batch."""
        # Define a function to process a batch
        def process_batch(batch):
            return [x * x for x in batch]
        
        # Process items in a single batch
        result = batch_process(
            [1, 2, 3],
            process_batch,
            batch_size=5,  # Larger than the input list
            parallel=True,
        )
        
        # Check result
        assert sorted(result) == [1, 4, 9]


class TestMonitorResourcesDecorator:
    """Tests for monitor_resources decorator."""

    @patch("crowd_sentiment_music_generator.utils.parallel_processing.ResourceMonitor")
    def test_decorator(self, mock_monitor_class):
        """Test monitor_resources decorator."""
        # Set up mock
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor
        mock_monitor.stop.return_value = {"duration": 1.0, "cpu": {"avg": 10.0}, "memory": {"avg": 30.0}}
        
        # Define a function with the decorator
        @monitor_resources
        def test_function(x, y):
            return x + y
        
        # Call the function
        result = test_function(2, 3)
        
        # Check result
        assert result == 5
        
        # Check that monitor was used
        mock_monitor.start.assert_called_once()
        mock_monitor.stop.assert_called_once()


@pytest.mark.performance
class TestPerformance:
    """Performance tests for parallel processing utilities."""

    def test_parallel_vs_sequential(self):
        """Compare parallel and sequential processing performance."""
        # Create test data
        data_size = 10
        data = list(range(data_size))
        
        # Define processing function
        def process_item(item):
            # Simulate CPU-intensive work
            time.sleep(0.1)
            return item * item
        
        # Measure sequential processing time
        start_time = time.time()
        sequential_results = [process_item(item) for item in data]
        sequential_time = time.time() - start_time
        
        # Measure parallel processing time
        start_time = time.time()
        parallel_results = parallel_map(process_item, data, max_workers=os.cpu_count())
        parallel_time = time.time() - start_time
        
        # Check that results are the same (after sorting)
        assert sorted(sequential_results) == sorted(parallel_results)
        
        # Log performance results
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis

    def test_batch_size_impact(self):
        """Test the impact of batch size on processing performance."""
        # Create test data
        data_size = 20
        data = list(range(data_size))
        
        # Define batch processing function
        def process_batch(batch):
            # Simulate batch processing
            time.sleep(0.1 * len(batch))
            return [item * item for item in batch]
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5, 10, 20]
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