"""Optimized music-video synchronization module with parallel processing capabilities.

This module provides an optimized version of the music-video synchronization
module with parallel processing capabilities for improved performance when
synchronizing multiple highlights.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union

from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.services.highlight_generator.music_video_sync import (
    MusicVideoSynchronizer,
    SyncPoint,
)
from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine
from crowd_sentiment_music_generator.utils.parallel_processing import (
    parallel_map,
    batch_process,
    monitor_resources,
    ResourceMonitor,
)
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class OptimizedMusicVideoSynchronizer(MusicVideoSynchronizer):
    """Optimized music-video synchronizer with parallel processing capabilities.
    
    This class extends the base MusicVideoSynchronizer with parallel processing
    capabilities for improved performance when synchronizing multiple highlights.
    
    Attributes:
        max_workers: Maximum number of worker processes/threads
        batch_size: Number of highlights per batch
    """
    
    def __init__(
        self,
        music_engine: Optional[MagentaMusicEngine] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 5,
    ):
        """Initialize optimized music-video synchronizer.
        
        Args:
            music_engine: Optional Magenta music engine for generating music
            max_workers: Maximum number of worker processes/threads (default: CPU count)
            batch_size: Number of highlights per batch
        """
        super().__init__(music_engine)
        self.max_workers = max_workers or os.cpu_count() or 4
        self.batch_size = batch_size
    
    @with_error_handling
    @monitor_resources
    def align_batch(
        self,
        music_dict: Dict[str, HighlightMusic],
        segment_dict: Dict[str, HighlightSegment],
    ) -> Dict[str, HighlightMusic]:
        """Align multiple music compositions to video segments in parallel.
        
        Args:
            music_dict: Dictionary mapping highlight IDs to music compositions
            segment_dict: Dictionary mapping highlight IDs to video segments
            
        Returns:
            Dictionary mapping highlight IDs to aligned music compositions
            
        Raises:
            ValueError: If highlight IDs don't match
        """
        self.logger.info(f"Aligning batch of {len(music_dict)} music compositions")
        
        # Create list of (music, segment) pairs for processing
        pairs = []
        for highlight_id, music in music_dict.items():
            segment = segment_dict.get(highlight_id)
            if segment:
                pairs.append((music, segment))
            else:
                self.logger.warning(f"Segment not found for highlight {highlight_id}")
        
        # Define a function to process a single pair
        def process_pair(pair: Tuple[HighlightMusic, HighlightSegment]) -> Tuple[str, HighlightMusic]:
            music, segment = pair
            aligned_music = self.align_music_to_video(music, segment)
            return segment.id, aligned_music
        
        # Process pairs in parallel
        results = parallel_map(
            process_pair,
            pairs,
            max_workers=self.max_workers,
            use_processes=True,  # Use processes for CPU-bound tasks
        )
        
        # Convert results to dictionary
        result_dict = dict(results)
        
        self.logger.info(f"Completed aligning {len(result_dict)} music compositions")
        return result_dict
    
    @with_error_handling
    @monitor_resources
    def adjust_duration_batch(
        self,
        music_dict: Dict[str, HighlightMusic],
        target_durations: Dict[str, float],
    ) -> Dict[str, HighlightMusic]:
        """Adjust durations of multiple music compositions in parallel.
        
        Args:
            music_dict: Dictionary mapping highlight IDs to music compositions
            target_durations: Dictionary mapping highlight IDs to target durations
            
        Returns:
            Dictionary mapping highlight IDs to adjusted music compositions
        """
        self.logger.info(f"Adjusting durations for batch of {len(music_dict)} music compositions")
        
        # Create list of (music, duration) pairs for processing
        pairs = []
        for highlight_id, music in music_dict.items():
            duration = target_durations.get(highlight_id)
            if duration is not None:
                pairs.append((highlight_id, music, duration))
            else:
                self.logger.warning(f"Target duration not found for highlight {highlight_id}")
        
        # Define a function to process a batch of pairs
        def process_batch(batch: List[Tuple[str, HighlightMusic, float]]) -> List[Tuple[str, HighlightMusic]]:
            results = []
            for highlight_id, music, duration in batch:
                adjusted_music = self._adjust_duration_smart(music, duration)
                results.append((highlight_id, adjusted_music))
            return results
        
        # Process pairs in batches
        batch_results = batch_process(
            pairs,
            process_batch,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            parallel=True,
        )
        
        # Convert results to dictionary
        result_dict = dict(batch_results)
        
        self.logger.info(f"Completed adjusting durations for {len(result_dict)} music compositions")
        return result_dict
    
    @with_error_handling
    @monitor_resources
    def generate_transitions_batch(
        self, music_dict: Dict[str, HighlightMusic]
    ) -> Dict[str, HighlightMusic]:
        """Generate transitions for multiple music compositions in parallel.
        
        Args:
            music_dict: Dictionary mapping highlight IDs to music compositions
            
        Returns:
            Dictionary mapping highlight IDs to music compositions with transitions
        """
        self.logger.info(f"Generating transitions for batch of {len(music_dict)} music compositions")
        
        # Define a function to process a single music composition
        def process_music(item: Tuple[str, HighlightMusic]) -> Tuple[str, HighlightMusic]:
            highlight_id, music = item
            music_with_transitions = self.generate_transitions(music)
            return highlight_id, music_with_transitions
        
        # Process music compositions in parallel
        results = parallel_map(
            process_music,
            list(music_dict.items()),
            max_workers=self.max_workers,
            use_processes=True,  # Use processes for CPU-bound tasks
        )
        
        # Convert results to dictionary
        result_dict = dict(results)
        
        self.logger.info(f"Completed generating transitions for {len(result_dict)} music compositions")
        return result_dict
    
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