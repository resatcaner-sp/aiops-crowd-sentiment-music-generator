"""Parallel processing utilities for audio analysis and batch processing.

This module provides utilities for parallel processing of audio data,
including concurrent audio analysis and batch processing for highlights.
"""

import concurrent.futures
import logging
import multiprocessing
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ResourceMonitor:
    """Monitor system resource usage during processing.
    
    This class provides methods to track CPU, memory, and disk usage
    during audio processing operations.
    
    Attributes:
        start_time: Time when monitoring started
        start_cpu_percent: CPU usage percentage at start
        start_memory_percent: Memory usage percentage at start
        start_disk_io: Disk I/O counters at start
        samples: List of resource usage samples
        sampling_interval: Interval between samples in seconds
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """Initialize resource monitor.
        
        Args:
            sampling_interval: Interval between samples in seconds
        """
        self.start_time = 0.0
        self.start_cpu_percent = 0.0
        self.start_memory_percent = 0.0
        self.start_disk_io = None
        self.samples = []
        self.sampling_interval = sampling_interval
        self._stop_sampling = False
        self._sampling_thread = None
    
    def start(self) -> None:
        """Start monitoring resource usage."""
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.start_memory_percent = psutil.virtual_memory().percent
        self.start_disk_io = psutil.disk_io_counters()
        self.samples = []
        self._stop_sampling = False
        
        # Start sampling in a separate thread
        self._sampling_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._sampling_thread.submit(self._sample_resources)
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return resource usage statistics.
        
        Returns:
            Dictionary with resource usage statistics
        """
        self._stop_sampling = True
        if self._sampling_thread:
            self._sampling_thread.shutdown(wait=True)
        
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        end_memory_percent = psutil.virtual_memory().percent
        end_disk_io = psutil.disk_io_counters()
        
        # Calculate statistics
        duration = end_time - self.start_time
        
        # CPU statistics
        cpu_samples = [sample["cpu_percent"] for sample in self.samples]
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        
        # Memory statistics
        memory_samples = [sample["memory_percent"] for sample in self.samples]
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        max_memory = max(memory_samples) if memory_samples else 0
        
        # Disk I/O statistics
        if self.start_disk_io and end_disk_io:
            read_bytes = end_disk_io.read_bytes - self.start_disk_io.read_bytes
            write_bytes = end_disk_io.write_bytes - self.start_disk_io.write_bytes
            read_count = end_disk_io.read_count - self.start_disk_io.read_count
            write_count = end_disk_io.write_count - self.start_disk_io.write_count
        else:
            read_bytes = write_bytes = read_count = write_count = 0
        
        return {
            "duration": duration,
            "cpu": {
                "start": self.start_cpu_percent,
                "end": end_cpu_percent,
                "avg": avg_cpu,
                "max": max_cpu,
            },
            "memory": {
                "start": self.start_memory_percent,
                "end": end_memory_percent,
                "avg": avg_memory,
                "max": max_memory,
            },
            "disk_io": {
                "read_bytes": read_bytes,
                "write_bytes": write_bytes,
                "read_count": read_count,
                "write_count": write_count,
            },
            "samples": self.samples,
        }
    
    def _sample_resources(self) -> None:
        """Sample resource usage at regular intervals."""
        while not self._stop_sampling:
            try:
                sample = {
                    "timestamp": time.time() - self.start_time,
                    "cpu_percent": psutil.cpu_percent(interval=0),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
                }
                self.samples.append(sample)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error sampling resources: {e}")
                break


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: Optional[int] = None,
    chunk_size: int = 1,
    use_processes: bool = True,
) -> List[R]:
    """Process items in parallel using a map pattern.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers (default: CPU count)
        chunk_size: Number of items per worker
        use_processes: Use processes instead of threads
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks
    executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = future_to_item[future]
                logger.error(f"Error processing item {item}: {e}")
                # Re-raise the exception if needed
                # raise
    
    return results


def batch_process(
    items: List[T],
    process_func: Callable[[List[T]], List[R]],
    batch_size: int = 10,
    max_workers: Optional[int] = None,
    parallel: bool = True,
) -> List[R]:
    """Process items in batches, optionally in parallel.
    
    Args:
        items: List of items to process
        process_func: Function to process a batch of items
        batch_size: Number of items per batch
        max_workers: Maximum number of workers for parallel processing
        parallel: Whether to process batches in parallel
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    if parallel and len(batches) > 1:
        # Process batches in parallel
        batch_results = parallel_map(process_func, batches, max_workers=max_workers)
        # Flatten results
        return [item for batch in batch_results for item in batch]
    else:
        # Process batches sequentially
        results = []
        for batch in batches:
            batch_result = process_func(batch)
            results.extend(batch_result)
        return results


def monitor_resources(func: Callable) -> Callable:
    """Decorator to monitor resource usage during function execution.
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            stats = monitor.stop()
            logger.info(
                f"Resource usage for {func.__name__}: "
                f"Duration={stats['duration']:.2f}s, "
                f"Avg CPU={stats['cpu']['avg']:.1f}%, "
                f"Max CPU={stats['cpu']['max']:.1f}%, "
                f"Avg Memory={stats['memory']['avg']:.1f}%, "
                f"Max Memory={stats['memory']['max']:.1f}%"
            )
    
    return wrapper