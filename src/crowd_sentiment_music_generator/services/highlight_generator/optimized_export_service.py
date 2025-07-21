"""Optimized export service for highlight music generator.

This module provides an optimized version of the export service with
parallel processing capabilities for improved performance when exporting
multiple highlights.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union

from pydantic import BaseModel

from crowd_sentiment_music_generator.config.cache_config import get_cache_settings
from crowd_sentiment_music_generator.exceptions.export_error import ExportError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.highlight_generator.export_service import (
    ExportService,
    ExportFormat,
    QualityPreset,
    ExportOptions,
    ExportMetadata,
)
from crowd_sentiment_music_generator.utils.parallel_processing import (
    parallel_map,
    batch_process,
    monitor_resources,
    ResourceMonitor,
)
from crowd_sentiment_music_generator.utils.cache import CacheManager
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class BatchExportRequest(BaseModel):
    """Request model for batch export.
    
    Attributes:
        highlight_id: ID of the highlight
        output_path: Path to save the exported file
        options: Export options
    """
    
    highlight_id: str
    output_path: str
    options: ExportOptions


class BatchExportResult(BaseModel):
    """Result model for batch export.
    
    Attributes:
        highlight_id: ID of the highlight
        output_path: Path to the exported file
        format: Export format
        quality: Quality preset
        success: Whether the export was successful
        error: Error message if export failed
    """
    
    highlight_id: str
    output_path: str
    format: str
    quality: str
    success: bool
    error: Optional[str] = None


class OptimizedExportService(ExportService):
    """Optimized export service with parallel processing capabilities.
    
    This class extends the base ExportService with parallel processing
    capabilities for improved performance when exporting multiple highlights.
    
    Attributes:
        max_workers: Maximum number of worker processes/threads
        batch_size: Number of highlights per batch
    """
    
    def __init__(self, max_workers: Optional[int] = None, batch_size: int = 5):
        """Initialize optimized export service.
        
        Args:
            max_workers: Maximum number of worker processes/threads (default: CPU count)
            batch_size: Number of highlights per batch
        """
        super().__init__()
        self.max_workers = max_workers or os.cpu_count() or 4
        self.batch_size = batch_size
        self.resource_monitor = ResourceMonitor()
    
    @with_error_handling
    @monitor_resources
    def export_highlight_batch(
        self,
        highlights: Dict[str, HighlightSegment],
        music: Dict[str, HighlightMusic],
        requests: List[BatchExportRequest],
    ) -> List[BatchExportResult]:
        """Export multiple highlights in parallel.
        
        Args:
            highlights: Dictionary mapping highlight IDs to highlight segments
            music: Dictionary mapping highlight IDs to highlight music
            requests: List of export requests
            
        Returns:
            List of export results
            
        Raises:
            ExportError: If batch export fails
        """
        self.logger.info(f"Exporting batch of {len(requests)} highlights")
        
        # Define a function to process a single export request
        def process_request(request: BatchExportRequest) -> BatchExportResult:
            highlight_id = request.highlight_id
            output_path = request.output_path
            options = request.options
            
            # Get highlight segment and music
            highlight = highlights.get(highlight_id)
            highlight_music = music.get(highlight_id)
            
            if not highlight or not highlight_music:
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=output_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=False,
                    error="Highlight or music not found",
                )
            
            try:
                # Export highlight
                exported_path = self.export_highlight(
                    highlight, highlight_music, output_path, options
                )
                
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=exported_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=True,
                )
            except Exception as e:
                self.logger.error(f"Error exporting highlight {highlight_id}: {str(e)}")
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=output_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=False,
                    error=str(e),
                )
        
        # Define a function to process a batch of requests
        def process_batch(batch: List[BatchExportRequest]) -> List[BatchExportResult]:
            return [process_request(request) for request in batch]
        
        # Process requests in batches
        results = batch_process(
            requests,
            process_batch,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            parallel=True,
        )
        
        # Log results
        successful = sum(1 for result in results if result.success)
        failed = len(results) - successful
        self.logger.info(
            f"Batch export completed: {successful} successful, {failed} failed"
        )
        
        return results
    
    @with_error_handling
    @monitor_resources
    def export_audio_batch(
        self,
        music: Dict[str, HighlightMusic],
        requests: List[BatchExportRequest],
    ) -> List[BatchExportResult]:
        """Export multiple audio files in parallel.
        
        Args:
            music: Dictionary mapping highlight IDs to highlight music
            requests: List of export requests
            
        Returns:
            List of export results
            
        Raises:
            ExportError: If batch export fails
        """
        self.logger.info(f"Exporting batch of {len(requests)} audio files")
        
        # Define a function to process a single export request
        def process_request(request: BatchExportRequest) -> BatchExportResult:
            highlight_id = request.highlight_id
            output_path = request.output_path
            options = request.options
            
            # Get highlight music
            highlight_music = music.get(highlight_id)
            
            if not highlight_music:
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=output_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=False,
                    error="Music not found",
                )
            
            try:
                # Export audio
                exported_path = self.export_audio_only(
                    highlight_music, output_path, options
                )
                
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=exported_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=True,
                )
            except Exception as e:
                self.logger.error(f"Error exporting audio for highlight {highlight_id}: {str(e)}")
                return BatchExportResult(
                    highlight_id=highlight_id,
                    output_path=output_path,
                    format=options.format.value,
                    quality=options.quality_preset.value,
                    success=False,
                    error=str(e),
                )
        
        # Process requests in parallel
        results = parallel_map(
            process_request,
            requests,
            max_workers=self.max_workers,
            use_processes=False,  # Use threads for I/O-bound tasks
        )
        
        # Log results
        successful = sum(1 for result in results if result.success)
        failed = len(results) - successful
        self.logger.info(
            f"Batch audio export completed: {successful} successful, {failed} failed"
        )
        
        return results
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage statistics.
        
        Returns:
            Dictionary with resource usage statistics
        """
        # In a real implementation, this would return actual resource usage statistics
        # For this implementation, we'll return simulated statistics
        return {
            "cpu_percent": 50.0,
            "memory_percent": 30.0,
            "disk_io": {
                "read_bytes": 1024 * 1024 * 10,  # 10 MB
                "write_bytes": 1024 * 1024 * 20,  # 20 MB
            },
        }
    
    def optimize_batch_size(self, sample_size: int = 5) -> int:
        """Optimize batch size based on system resources.
        
        Args:
            sample_size: Number of samples to use for optimization
            
        Returns:
            Optimal batch size
        """
        # In a real implementation, this would run tests with different batch sizes
        # and measure performance to determine the optimal batch size
        # For this implementation, we'll use a simple heuristic
        
        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        
        # Simple heuristic: 1 batch item per CPU core, adjusted for available memory
        optimal_batch_size = max(1, min(cpu_count, int(memory_gb / 2)))
        
        self.logger.info(
            f"Optimized batch size: {optimal_batch_size} "
            f"(CPU cores: {cpu_count}, Memory: {memory_gb:.1f} GB)"
        )
        
        return optimal_batch_size