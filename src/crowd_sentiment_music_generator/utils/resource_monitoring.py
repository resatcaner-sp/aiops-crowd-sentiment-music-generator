"""Resource monitoring utilities for tracking system resource usage.

This module provides utilities for monitoring system resource usage,
including CPU, memory, and disk I/O, during audio processing operations.
"""

import logging
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import psutil

logger = logging.getLogger(__name__)


class ResourceUsageStats:
    """Statistics for resource usage monitoring.
    
    Attributes:
        start_time: Start time of monitoring
        end_time: End time of monitoring
        duration: Duration of monitoring in seconds
        cpu_stats: CPU usage statistics
        memory_stats: Memory usage statistics
        disk_stats: Disk I/O statistics
        samples: List of resource usage samples
    """
    
    def __init__(self):
        """Initialize resource usage statistics."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0
        self.cpu_stats = {
            "start": 0.0,
            "end": 0.0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
        }
        self.memory_stats = {
            "start": 0.0,
            "end": 0.0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
        }
        self.disk_stats = {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
        }
        self.samples = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary.
        
        Returns:
            Dictionary representation of statistics
        """
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "cpu": self.cpu_stats,
            "memory": self.memory_stats,
            "disk_io": self.disk_stats,
            "samples": self.samples,
        }
    
    def __str__(self) -> str:
        """Get string representation of statistics.
        
        Returns:
            String representation
        """
        return (
            f"Duration: {self.duration:.2f}s, "
            f"CPU: {self.cpu_stats['avg']:.1f}% (max: {self.cpu_stats['max']:.1f}%), "
            f"Memory: {self.memory_stats['avg']:.1f}% (max: {self.memory_stats['max']:.1f}%)"
        )


class ResourceMonitor:
    """Monitor system resource usage.
    
    This class provides methods to track CPU, memory, and disk usage
    during operations.
    
    Attributes:
        sampling_interval: Interval between samples in seconds
        stats: Resource usage statistics
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """Initialize resource monitor.
        
        Args:
            sampling_interval: Interval between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.stats = ResourceUsageStats()
        self._stop_sampling = False
        self._sampling_thread = None
    
    def start(self) -> None:
        """Start monitoring resource usage."""
        self.stats = ResourceUsageStats()
        self.stats.start_time = time.time()
        self.stats.cpu_stats["start"] = psutil.cpu_percent(interval=0.1)
        self.stats.memory_stats["start"] = psutil.virtual_memory().percent
        self.stats.disk_stats = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        self.stats.samples = []
        self._stop_sampling = False
        
        # Start sampling in a separate thread
        import threading
        self._sampling_thread = threading.Thread(target=self._sample_resources)
        self._sampling_thread.daemon = True
        self._sampling_thread.start()
    
    def stop(self) -> ResourceUsageStats:
        """Stop monitoring and return resource usage statistics.
        
        Returns:
            Resource usage statistics
        """
        self._stop_sampling = True
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2.0)
        
        self.stats.end_time = time.time()
        self.stats.duration = self.stats.end_time - self.stats.start_time
        self.stats.cpu_stats["end"] = psutil.cpu_percent(interval=0.1)
        self.stats.memory_stats["end"] = psutil.virtual_memory().percent
        
        end_disk_io = psutil.disk_io_counters()
        if end_disk_io and self.stats.disk_stats:
            start_disk_io = self.stats.disk_stats
            self.stats.disk_stats = {
                "read_bytes": end_disk_io.read_bytes - start_disk_io.get("read_bytes", 0),
                "write_bytes": end_disk_io.write_bytes - start_disk_io.get("write_bytes", 0),
                "read_count": end_disk_io.read_count - start_disk_io.get("read_count", 0),
                "write_count": end_disk_io.write_count - start_disk_io.get("write_count", 0),
            }
        
        # Calculate statistics from samples
        if self.stats.samples:
            cpu_values = [sample["cpu_percent"] for sample in self.stats.samples]
            memory_values = [sample["memory_percent"] for sample in self.stats.samples]
            
            self.stats.cpu_stats["min"] = min(cpu_values)
            self.stats.cpu_stats["max"] = max(cpu_values)
            self.stats.cpu_stats["avg"] = sum(cpu_values) / len(cpu_values)
            
            self.stats.memory_stats["min"] = min(memory_values)
            self.stats.memory_stats["max"] = max(memory_values)
            self.stats.memory_stats["avg"] = sum(memory_values) / len(memory_values)
        
        return self.stats
    
    def _sample_resources(self) -> None:
        """Sample resource usage at regular intervals."""
        while not self._stop_sampling:
            try:
                sample = {
                    "timestamp": time.time() - self.stats.start_time,
                    "cpu_percent": psutil.cpu_percent(interval=0),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
                }
                self.stats.samples.append(sample)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error sampling resources: {e}")
                break


class ResourceUsageLogger:
    """Logger for resource usage statistics.
    
    This class provides methods to log resource usage statistics
    to a file or database.
    
    Attributes:
        log_file: Path to log file
        log_to_file: Whether to log to file
        log_to_db: Whether to log to database
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_to_file: bool = True,
        log_to_db: bool = False,
    ):
        """Initialize resource usage logger.
        
        Args:
            log_file: Path to log file
            log_to_file: Whether to log to file
            log_to_db: Whether to log to database
        """
        self.log_file = log_file or "resource_usage.log"
        self.log_to_file = log_to_file
        self.log_to_db = log_to_db
        self.logger = logging.getLogger(__name__)
    
    def log_usage(
        self, operation: str, stats: ResourceUsageStats, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log resource usage statistics.
        
        Args:
            operation: Operation name
            stats: Resource usage statistics
            details: Additional details
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "duration": stats.duration,
            "cpu": stats.cpu_stats,
            "memory": stats.memory_stats,
            "disk_io": stats.disk_stats,
            "details": details or {},
        }
        
        # Log to console
        self.logger.info(
            f"{operation}: {stats.duration:.2f}s, "
            f"CPU: {stats.cpu_stats['avg']:.1f}%, "
            f"Memory: {stats.memory_stats['avg']:.1f}%"
        )
        
        # Log to file
        if self.log_to_file:
            self._log_to_file(log_entry)
        
        # Log to database
        if self.log_to_db:
            self._log_to_db(log_entry)
    
    def _log_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Log entry to file.
        
        Args:
            log_entry: Log entry
        """
        try:
            import json
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Append to log file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Error logging to file: {e}")
    
    def _log_to_db(self, log_entry: Dict[str, Any]) -> None:
        """Log entry to database.
        
        Args:
            log_entry: Log entry
        """
        # In a real implementation, this would log to a database
        # For this implementation, we'll just log a message
        self.logger.info("Database logging not implemented")


def monitor_resources(operation: Optional[str] = None):
    """Decorator to monitor resource usage during function execution.
    
    Args:
        operation: Operation name (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get operation name
            op_name = operation or func.__name__
            
            # Create resource monitor
            monitor = ResourceMonitor()
            monitor.start()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop monitoring and log statistics
                stats = monitor.stop()
                logger.info(
                    f"Resource usage for {op_name}: {stats}"
                )
        
        return wrapper
    
    # Handle case where decorator is used without arguments
    if callable(operation):
        func = operation
        operation = None
        return decorator(func)
    
    return decorator


class SystemResourceMonitor:
    """Monitor system-wide resource usage.
    
    This class provides methods to monitor system-wide resource usage,
    including CPU, memory, and disk I/O.
    
    Attributes:
        sampling_interval: Interval between samples in seconds
        max_samples: Maximum number of samples to keep
    """
    
    _instance = None
    
    def __new__(cls, *args: Any, **kwargs: Any) -> "SystemResourceMonitor":
        """Create a singleton instance of SystemResourceMonitor."""
        if cls._instance is None:
            cls._instance = super(SystemResourceMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, sampling_interval: float = 5.0, max_samples: int = 100):
        """Initialize system resource monitor.
        
        Args:
            sampling_interval: Interval between samples in seconds
            max_samples: Maximum number of samples to keep
        """
        # Only initialize once
        if hasattr(self, "initialized"):
            return
        
        self.sampling_interval = sampling_interval
        self.max_samples = max_samples
        self.samples = []
        self._stop_sampling = False
        self._sampling_thread = None
        self.initialized = True
    
    def start(self) -> None:
        """Start monitoring system resource usage."""
        if self._sampling_thread is not None:
            return
        
        self._stop_sampling = False
        
        # Start sampling in a separate thread
        import threading
        self._sampling_thread = threading.Thread(target=self._sample_resources)
        self._sampling_thread.daemon = True
        self._sampling_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring system resource usage."""
        self._stop_sampling = True
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2.0)
            self._sampling_thread = None
    
    def get_samples(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get resource usage samples.
        
        Args:
            count: Number of most recent samples to return
            
        Returns:
            List of resource usage samples
        """
        if count is None:
            return self.samples
        return self.samples[-count:]
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage.
        
        Returns:
            Dictionary with current resource usage
        """
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
        }
    
    def _sample_resources(self) -> None:
        """Sample resource usage at regular intervals."""
        while not self._stop_sampling:
            try:
                sample = self.get_current_usage()
                
                # Add sample to list
                self.samples.append(sample)
                
                # Limit number of samples
                if len(self.samples) > self.max_samples:
                    self.samples = self.samples[-self.max_samples:]
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error sampling system resources: {e}")
                break