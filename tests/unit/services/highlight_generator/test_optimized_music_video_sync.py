"""Unit tests for optimized music-video synchronization."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic, MusicSegment
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.highlight_generator.optimized_music_video_sync import (
    OptimizedMusicVideoSynchronizer,
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
def sample_highlight_segments(tmp_path):
    """Create sample highlight segments for testing."""
    segments = {}
    for i in range(5):
        # Create dummy video file
        video_path = tmp_path / f"test_video_{i}.mp4"
        with open(video_path, "w") as f:
            f.write(f"Dummy video content {i}")
        
        # Create highlight segment
        segment = HighlightSegment(
            id=f"highlight_{i}",
            start_time=i * 30.0,
            end_time=(i + 1) * 30.0,
            key_moment_time=i * 30.0 + 15.0,
            video_path=str(video_path),
            events=[]
        )
        segments[f"highlight_{i}"] = segment
    
    return segments


@pytest.fixture
def sample_highlight_music(sample_parameters):
    """Create sample highlight music for testing."""
    music_dict = {}
    for i in range(5):
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
        
        music = HighlightMusic(
            highlight_id=f"highlight_{i}",
            segments=segments,
            base_parameters=sample_parameters,
            duration=30.0
        )
        music_dict[f"highlight_{i}"] = music
    
    return music_dict


class TestOptimizedMusicVideoSynchronizer:
    """Tests for OptimizedMusicVideoSynchronizer."""

    @pytest.fixture
    def synchronizer(self):
        """Create an optimized music-video synchronizer."""
        return OptimizedMusicVideoSynchronizer(max_workers=2, batch_size=2)

    def test_align_batch(self, synchronizer, sample_highlight_music, sample_highlight_segments):
        """Test aligning a batch of music compositions."""
        # Mock align_music_to_video to avoid actual alignment
        with patch.object(synchronizer, "align_music_to_video") as mock_align:
            # Set up mock
            mock_align.side_effect = lambda music, segment: music
            
            # Align batch
            results = synchronizer.align_batch(sample_highlight_music, sample_highlight_segments)
            
            # Check results
            assert len(results) == len(sample_highlight_music)
            assert all(highlight_id in results for highlight_id in sample_highlight_music.keys())
            assert all(isinstance(music, HighlightMusic) for music in results.values())
            
            # Check that align_music_to_video was called for each music composition
            assert mock_align.call_count == len(sample_highlight_music)

    def test_adjust_duration_batch(self, synchronizer, sample_highlight_music):
        """Test adjusting durations of a batch of music compositions."""
        # Create target durations
        target_durations = {
            highlight_id: 45.0 for highlight_id in sample_highlight_music.keys()
        }
        
        # Mock _adjust_duration_smart to avoid actual adjustment
        with patch.object(synchronizer, "_adjust_duration_smart") as mock_adjust:
            # Set up mock
            mock_adjust.side_effect = lambda music, duration: music
            
            # Adjust durations
            results = synchronizer.adjust_duration_batch(sample_highlight_music, target_durations)
            
            # Check results
            assert len(results) == len(sample_highlight_music)
            assert all(highlight_id in results for highlight_id in sample_highlight_music.keys())
            assert all(isinstance(music, HighlightMusic) for music in results.values())
            
            # Check that _adjust_duration_smart was called for each music composition
            assert mock_adjust.call_count == len(sample_highlight_music)

    def test_generate_transitions_batch(self, synchronizer, sample_highlight_music):
        """Test generating transitions for a batch of music compositions."""
        # Mock generate_transitions to avoid actual transition generation
        with patch.object(synchronizer, "generate_transitions") as mock_generate:
            # Set up mock
            mock_generate.side_effect = lambda music: music
            
            # Generate transitions
            results = synchronizer.generate_transitions_batch(sample_highlight_music)
            
            # Check results
            assert len(results) == len(sample_highlight_music)
            assert all(highlight_id in results for highlight_id in sample_highlight_music.keys())
            assert all(isinstance(music, HighlightMusic) for music in results.values())
            
            # Check that generate_transitions was called for each music composition
            assert mock_generate.call_count == len(sample_highlight_music)

    def test_optimize_batch_size(self, synchronizer):
        """Test optimizing batch size."""
        # Optimize batch size
        batch_size = synchronizer.optimize_batch_size()
        
        # Check result
        assert batch_size > 0
        assert isinstance(batch_size, int)


@pytest.mark.performance
class TestMusicVideoSyncPerformance:
    """Performance tests for optimized music-video synchronization."""

    @pytest.fixture
    def synchronizer(self):
        """Create an optimized music-video synchronizer."""
        return OptimizedMusicVideoSynchronizer(max_workers=os.cpu_count(), batch_size=2)

    @pytest.fixture
    def sequential_synchronizer(self):
        """Create a sequential music-video synchronizer."""
        return OptimizedMusicVideoSynchronizer(max_workers=1, batch_size=1)

    @pytest.fixture
    def large_highlight_music(self, sample_parameters):
        """Create a large set of highlight music for testing."""
        music_dict = {}
        for i in range(10):
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
            
            music = HighlightMusic(
                highlight_id=f"highlight_{i}",
                segments=segments,
                base_parameters=sample_parameters,
                duration=30.0
            )
            music_dict[f"highlight_{i}"] = music
        
        return music_dict

    @pytest.fixture
    def large_highlight_segments(self, tmp_path):
        """Create a large set of highlight segments for testing."""
        segments = {}
        for i in range(10):
            # Create dummy video file
            video_path = tmp_path / f"test_video_{i}.mp4"
            with open(video_path, "w") as f:
                f.write(f"Dummy video content {i}")
            
            # Create highlight segment
            segment = HighlightSegment(
                id=f"highlight_{i}",
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                key_moment_time=i * 30.0 + 15.0,
                video_path=str(video_path),
                events=[]
            )
            segments[f"highlight_{i}"] = segment
        
        return segments

    def test_parallel_vs_sequential_align(self, synchronizer, sequential_synchronizer, large_highlight_music, large_highlight_segments):
        """Compare parallel and sequential alignment performance."""
        # Mock align_music_to_video to simulate alignment with delay
        def mock_align(music, segment):
            # Simulate alignment with delay
            time.sleep(0.1)
            return music
        
        # Test sequential alignment
        with patch.object(sequential_synchronizer, "align_music_to_video", side_effect=mock_align):
            start_time = time.time()
            sequential_results = sequential_synchronizer.align_batch(
                large_highlight_music, large_highlight_segments
            )
            sequential_time = time.time() - start_time
        
        # Test parallel alignment
        with patch.object(synchronizer, "align_music_to_video", side_effect=mock_align):
            start_time = time.time()
            parallel_results = synchronizer.align_batch(
                large_highlight_music, large_highlight_segments
            )
            parallel_time = time.time() - start_time
        
        # Check that results are the same
        assert len(sequential_results) == len(parallel_results)
        
        # Log performance results
        print(f"Sequential alignment time: {sequential_time:.2f}s")
        print(f"Parallel alignment time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis

    def test_batch_size_impact(self, large_highlight_music):
        """Test the impact of batch size on transition generation performance."""
        # Mock generate_transitions to simulate transition generation with delay
        def mock_generate(music):
            # Simulate transition generation with delay
            time.sleep(0.1)
            return music
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5, 10]
        times = []
        
        for batch_size in batch_sizes:
            # Create synchronizer with specific batch size
            synchronizer = OptimizedMusicVideoSynchronizer(max_workers=os.cpu_count(), batch_size=batch_size)
            
            # Measure transition generation time
            with patch.object(synchronizer, "generate_transitions", side_effect=mock_generate):
                start_time = time.time()
                results = synchronizer.generate_transitions_batch(large_highlight_music)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
            
            # Check results
            assert len(results) == len(large_highlight_music)
        
        # Log results for analysis
        for batch_size, elapsed_time in zip(batch_sizes, times):
            print(f"Batch size {batch_size}: {elapsed_time:.2f}s")
        
        # We don't assert specific performance characteristics, as they depend on the system
        # and the specific test environment, but we log the results for analysis