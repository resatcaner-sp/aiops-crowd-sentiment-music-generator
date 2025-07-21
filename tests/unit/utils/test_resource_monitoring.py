"""Unit tests for resource monitoring utilities."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.utils.resource_monitoring import (
    ResourceUsageStats,
    ResourceMonitor,
    ResourceUsageLogger,
    monitor_resources,
    SystemResourceMonitor,
)


class TestResourceUsageStats:
    """Tests for ResourceUsageStats."""

    def test_initialization(self):
        """Test initialization of ResourceUsageStats."""
        stats = ResourceUsageStats()
        
        # Check default values
        assert stats.start_time == 0.0
        assert stats.end_time == 0.0
        assert stats.duration == 0.0
        assert stats.cpu_stats["avg"] == 0.0
        assert stats.memory_stats["avg"] == 0.0
        assert stats.disk_stats["read_bytes"] == 0
        assert len(stats.samples) == 0

    def test_to_dict(self):
        """Test converting ResourceUsageStats to dictionary."""
        stats = ResourceUsageStats()
        stats.start_time = 100.0
        stats.end_time = 105.0
        stats.duration = 5.0
        stats.cpu_stats = {"start": 10.0, "end": 20.0, "min": 10.0, "max": 30.0, "avg": 20.0}
        stats.memory_stats = {"start": 30.0, "end": 40.0, "min": 30.0, "max": 50.0, "avg": 40.0}
        stats.disk_stats = {"read_bytes": 1000, "write_bytes": 2000, "read_count": 10, "write_count": 20}
        stats.samples = [{"timestamp": 1.0, "cpu_percent": 20.0, "memory_percent": 40.0}]
        
        # Convert to dictionary
        result = stats.to_dict()
        
        # Check result
        assert result["start_time"] == 100.0
        assert result["end_time"] == 105.0
        assert result["duration"] == 5.0
        assert result["cpu"]["avg"] == 20.0
        assert result["memory"]["avg"] == 40.0
        assert result["disk_io"]["read_bytes"] == 1000
        assert len(result["samples"]) == 1

    def test_str_representation(self):
        """Test string representation of ResourceUsageStats."""
        stats = ResourceUsageStats()
        stats.duration = 5.0
        stats.cpu_stats = {"avg": 20.0, "max": 30.0}
        stats.memory_stats = {"avg": 40.0, "max": 50.0}
        
        # Get string representation
        result = str(stats)
        
        # Check result
        assert "Duration: 5.00s" in result
        assert "CPU: 20.0%" in result
        assert "max: 30.0%" in result
        assert "Memory: 40.0%" in result
        assert "max: 50.0%" in result


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
        assert stats.start_time > 0
        assert stats.end_time > stats.start_time
        assert stats.duration > 0
        assert stats.cpu_stats["start"] == 10.0
        assert stats.memory_stats["start"] == 30.0
        assert len(stats.samples) > 0

    def test_monitor_with_context_manager(self, monitor):
        """Test using ResourceMonitor as a context manager."""
        # Mock _sample_resources to avoid actual sampling
        with patch.object(monitor, "_sample_resources"):
            # Start and stop monitoring
            monitor.start()
            stats = monitor.stop()
            
            # Check results
            assert stats.start_time > 0
            assert stats.end_time > stats.start_time
            assert stats.duration > 0


class TestResourceUsageLogger:
    """Tests for ResourceUsageLogger."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create a resource usage logger."""
        log_file = tmp_path / "resource_usage.log"
        return ResourceUsageLogger(log_file=str(log_file), log_to_file=True, log_to_db=False)

    def test_log_usage(self, logger, caplog):
        """Test logging resource usage."""
        # Create stats
        stats = ResourceUsageStats()
        stats.duration = 5.0
        stats.cpu_stats = {"avg": 20.0, "max": 30.0}
        stats.memory_stats = {"avg": 40.0, "max": 50.0}
        stats.disk_stats = {"read_bytes": 1000, "write_bytes": 2000, "read_count": 10, "write_count": 20}
        
        # Log usage
        logger.log_usage("test_operation", stats, {"param": "value"})
        
        # Check console log
        assert "test_operation" in caplog.text
        assert "5.00s" in caplog.text
        assert "20.0%" in caplog.text
        assert "40.0%" in caplog.text
        
        # Check file log
        with open(logger.log_file, "r") as f:
            log_content = f.read()
            assert "test_operation" in log_content
            assert "5.0" in log_content
            assert "20.0" in log_content
            assert "40.0" in log_content
            assert "param" in log_content
            assert "value" in log_content


class TestMonitorResourcesDecorator:
    """Tests for monitor_resources decorator."""

    @patch("crowd_sentiment_music_generator.utils.resource_monitoring.ResourceMonitor")
    def test_decorator(self, mock_monitor_class):
        """Test monitor_resources decorator."""
        # Set up mock
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor
        mock_monitor.stop.return_value = ResourceUsageStats()
        
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

    @patch("crowd_sentiment_music_generator.utils.resource_monitoring.ResourceMonitor")
    def test_decorator_with_name(self, mock_monitor_class):
        """Test monitor_resources decorator with operation name."""
        # Set up mock
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor
        mock_monitor.stop.return_value = ResourceUsageStats()
        
        # Define a function with the decorator
        @monitor_resources("custom_operation")
        def test_function(x, y):
            return x + y
        
        # Call the function
        result = test_function(2, 3)
        
        # Check result
        assert result == 5
        
        # Check that monitor was used
        mock_monitor.start.assert_called_once()
        mock_monitor.stop.assert_called_once()


class TestSystemResourceMonitor:
    """Tests for SystemResourceMonitor."""

    def test_singleton_pattern(self):
        """Test that SystemResourceMonitor is a singleton."""
        monitor1 = SystemResourceMonitor()
        monitor2 = SystemResourceMonitor()
        assert monitor1 is monitor2

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_io_counters")
    def test_get_current_usage(self, mock_disk_io, mock_memory, mock_cpu):
        """Test getting current resource usage."""
        # Set up mocks
        mock_cpu.return_value = 10.0
        mock_memory.return_value.percent = 30.0
        mock_disk_io.return_value._asdict.return_value = {
            "read_bytes": 1000,
            "write_bytes": 2000,
            "read_count": 10,
            "write_count": 20,
        }
        
        # Get current usage
        monitor = SystemResourceMonitor()
        usage = monitor.get_current_usage()
        
        # Check result
        assert usage["cpu_percent"] == 10.0
        assert usage["memory_percent"] == 30.0
        assert usage["disk_io"]["read_bytes"] == 1000

    def test_start_stop(self):
        """Test starting and stopping system resource monitoring."""
        # Create monitor
        monitor = SystemResourceMonitor(sampling_interval=0.1)
        
        # Mock _sample_resources to avoid actual sampling
        with patch.object(monitor, "_sample_resources"):
            # Start monitoring
            monitor.start()
            
            # Check that sampling thread was created
            assert monitor._sampling_thread is not None
            
            # Stop monitoring
            monitor.stop()
            
            # Check that sampling thread was stopped
            assert monitor._sampling_thread is None

    def test_get_samples(self):
        """Test getting resource usage samples."""
        # Create monitor
        monitor = SystemResourceMonitor()
        
        # Add some samples
        monitor.samples = [
            {"timestamp": 1.0, "cpu_percent": 10.0, "memory_percent": 30.0},
            {"timestamp": 2.0, "cpu_percent": 20.0, "memory_percent": 40.0},
            {"timestamp": 3.0, "cpu_percent": 30.0, "memory_percent": 50.0},
        ]
        
        # Get all samples
        all_samples = monitor.get_samples()
        assert len(all_samples) == 3
        
        # Get limited samples
        limited_samples = monitor.get_samples(2)
        assert len(limited_samples) == 2
        assert limited_samples[0]["cpu_percent"] == 20.0
        assert limited_samples[1]["cpu_percent"] == 30.0