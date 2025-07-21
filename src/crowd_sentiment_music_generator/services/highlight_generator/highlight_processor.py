"""Highlight processing module for analyzing video segments and extracting events."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel

from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic, MusicSegment
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.synchronization.sync_engine import SyncEngine
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class HighlightProcessor:
    """Processes video highlights to analyze content and extract events.
    
    This class provides methods for analyzing video segments, extracting events,
    and generating appropriate music compositions for fixed durations.
    
    Attributes:
        sync_engine: Synchronization engine for aligning timestamps
        logger: Logger instance
    """
    
    # Mapping of event types to musical importance
    EVENT_IMPORTANCE = {
        "goal": 1.0,
        "penalty": 0.9,
        "red_card": 0.8,
        "yellow_card": 0.6,
        "shot_on_target": 0.7,
        "shot_off_target": 0.5,
        "corner": 0.4,
        "free_kick": 0.5,
        "substitution": 0.3,
        "kickoff": 0.4,
        "halftime": 0.6,
        "fulltime": 0.7
    }
    
    def __init__(self, sync_engine: Optional[SyncEngine] = None):
        """Initialize highlight processor.
        
        Args:
            sync_engine: Optional synchronization engine for timestamp alignment
        """
        self.sync_engine = sync_engine
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    def process_highlight(self, segment: HighlightSegment) -> HighlightSegment:
        """Process a video highlight segment for analysis.
        
        Args:
            segment: Highlight segment to process
            
        Returns:
            Processed highlight segment with additional metadata
            
        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If segment has invalid timestamps
        """
        # Validate segment
        if segment.end_time <= segment.start_time:
            raise ValueError("End time must be after start time")
        
        if segment.key_moment_time < segment.start_time or segment.key_moment_time > segment.end_time:
            raise ValueError("Key moment time must be within segment duration")
        
        # Check if video file exists
        if not os.path.exists(segment.video_path):
            raise FileNotFoundError(f"Video file not found: {segment.video_path}")
        
        # Extract video metadata (in a real implementation, this would use a video processing library)
        # For now, we'll just add some dummy metadata
        metadata = segment.metadata or {}
        metadata.update({
            "processed": True,
            "duration": segment.duration,
            "frame_rate": 30.0,  # Dummy value
            "resolution": "1920x1080",  # Dummy value
            "has_audio": True  # Dummy value
        })
        
        # Update segment with metadata
        updated_segment = segment.copy(update={"metadata": metadata})
        
        self.logger.info(
            f"Processed highlight segment {segment.id} "
            f"({segment.duration:.2f}s) from {segment.video_path}"
        )
        
        return updated_segment
    
    @with_error_handling
    def extract_events(self, segment: HighlightSegment) -> List[MatchEvent]:
        """Extract events from a highlight segment.
        
        Args:
            segment: Highlight segment to extract events from
            
        Returns:
            List of match events in the segment
            
        Raises:
            SynchronizationError: If sync engine is not available or not synchronized
        """
        if not self.sync_engine:
            raise SynchronizationError("Sync engine not available")
        
        # Get events for the segment time window
        try:
            # Convert segment timestamps to reference timeline
            start_time = self.sync_engine.align_timestamp(
                segment.start_time, "video", "reference"
            )
            end_time = self.sync_engine.align_timestamp(
                segment.end_time, "video", "reference"
            )
            
            # Get events in time window
            events = self.sync_engine.event_buffer.get_events_in_window(
                start_time, end_time, mark_processed=False
            )
            
            # Update segment with event IDs
            event_ids = [event.id for event in events]
            
            self.logger.info(
                f"Extracted {len(events)} events for highlight segment {segment.id}"
            )
            
            return events
        
        except SynchronizationError as e:
            self.logger.error(f"Failed to extract events: {str(e)}")
            return []
    
    @with_error_handling
    def generate_music_composition(
        self, segment: HighlightSegment, events: List[MatchEvent]
    ) -> HighlightMusic:
        """Generate a music composition for a fixed duration highlight.
        
        Args:
            segment: Highlight segment to generate music for
            events: List of match events in the segment
            
        Returns:
            Highlight music composition
            
        Raises:
            ValueError: If segment has invalid timestamps
        """
        # Validate segment
        if segment.end_time <= segment.start_time:
            raise ValueError("End time must be after start time")
        
        # Determine base musical parameters based on events
        base_parameters = self._determine_base_parameters(events)
        
        # Create music segments based on events
        music_segments = self._create_music_segments(segment, events, base_parameters)
        
        # Create highlight music composition
        highlight_music = HighlightMusic(
            highlight_id=segment.id,
            segments=music_segments,
            base_parameters=base_parameters,
            duration=segment.duration
        )
        
        self.logger.info(
            f"Generated music composition for highlight segment {segment.id} "
            f"with {len(music_segments)} music segments"
        )
        
        return highlight_music
    
    def _determine_base_parameters(self, events: List[MatchEvent]) -> MusicalParameters:
        """Determine base musical parameters based on events.
        
        Args:
            events: List of match events
            
        Returns:
            Base musical parameters
        """
        # Default parameters for empty event list
        if not events:
            return MusicalParameters(
                tempo=100.0,
                key="C Major",
                intensity=0.5,
                instrumentation=["piano", "strings", "percussion"],
                mood="neutral"
            )
        
        # Calculate average intensity based on event importance
        total_importance = 0.0
        event_count = 0
        
        for event in events:
            importance = self.EVENT_IMPORTANCE.get(event.type, 0.2)
            total_importance += importance
            event_count += 1
        
        avg_intensity = total_importance / event_count if event_count > 0 else 0.5
        
        # Determine tempo based on intensity
        # Higher intensity = faster tempo
        tempo = 80.0 + (avg_intensity * 60.0)  # Range: 80-140 BPM
        
        # Determine key based on most important event
        most_important_event = max(
            events, 
            key=lambda e: self.EVENT_IMPORTANCE.get(e.type, 0.0)
        )
        
        # Map event types to keys (simplified)
        key_mapping = {
            "goal": "C Major",
            "penalty": "G Major",
            "red_card": "D Minor",
            "yellow_card": "A Minor",
            "shot_on_target": "F Major",
            "shot_off_target": "E Minor",
            "default": "C Major"
        }
        
        key = key_mapping.get(most_important_event.type, key_mapping["default"])
        
        # Determine mood based on event types
        positive_events = ["goal", "shot_on_target"]
        negative_events = ["red_card", "yellow_card"]
        tense_events = ["penalty", "free_kick"]
        
        positive_count = sum(1 for e in events if e.type in positive_events)
        negative_count = sum(1 for e in events if e.type in negative_events)
        tense_count = sum(1 for e in events if e.type in tense_events)
        
        if positive_count > negative_count and positive_count > tense_count:
            mood = "bright"
        elif negative_count > positive_count and negative_count > tense_count:
            mood = "dark"
        elif tense_count > positive_count and tense_count > negative_count:
            mood = "tense"
        else:
            mood = "neutral"
        
        # Determine instrumentation based on mood
        instrumentation_mapping = {
            "bright": ["piano", "strings", "brass", "percussion"],
            "dark": ["strings", "low_brass", "timpani"],
            "tense": ["strings", "percussion", "synth"],
            "neutral": ["piano", "strings", "light_percussion"]
        }
        
        instrumentation = instrumentation_mapping.get(mood, ["piano", "strings"])
        
        return MusicalParameters(
            tempo=tempo,
            key=key,
            intensity=avg_intensity,
            instrumentation=instrumentation,
            mood=mood
        )
    
    def _create_music_segments(
        self, 
        segment: HighlightSegment, 
        events: List[MatchEvent],
        base_parameters: MusicalParameters
    ) -> List[MusicSegment]:
        """Create music segments based on events.
        
        Args:
            segment: Highlight segment
            events: List of match events
            base_parameters: Base musical parameters
            
        Returns:
            List of music segments
        """
        # If no events, create a single segment for the entire duration
        if not events:
            return [
                MusicSegment(
                    start_time=0.0,
                    end_time=segment.duration,
                    parameters=base_parameters
                )
            ]
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Convert event timestamps to segment-relative time
        relative_events = []
        for event in sorted_events:
            # Calculate relative timestamp
            relative_time = event.timestamp - segment.start_time
            
            # Ensure it's within segment bounds
            if 0 <= relative_time <= segment.duration:
                relative_events.append((relative_time, event))
        
        # If no events within bounds, create a single segment
        if not relative_events:
            return [
                MusicSegment(
                    start_time=0.0,
                    end_time=segment.duration,
                    parameters=base_parameters
                )
            ]
        
        # Create segments based on events
        music_segments = []
        
        # Add intro segment if first event is not at the beginning
        first_event_time = relative_events[0][0]
        if first_event_time > 1.0:  # If first event is more than 1 second in
            intro_params = base_parameters.copy()
            intro_params.intensity = max(0.3, intro_params.intensity - 0.2)
            
            music_segments.append(
                MusicSegment(
                    start_time=0.0,
                    end_time=first_event_time,
                    parameters=intro_params,
                    transition_in=True,
                    transition_out=True
                )
            )
        
        # Process each event
        for i, (event_time, event) in enumerate(relative_events):
            # Determine segment end time
            if i < len(relative_events) - 1:
                next_event_time = relative_events[i + 1][0]
                end_time = next_event_time
            else:
                end_time = segment.duration
            
            # Skip if segment would be too short
            if end_time - event_time < 0.5:
                continue
            
            # Create parameters for this segment
            event_params = base_parameters.copy()
            
            # Adjust parameters based on event type
            importance = self.EVENT_IMPORTANCE.get(event.type, 0.2)
            
            # More important events get higher intensity
            event_params.intensity = min(1.0, base_parameters.intensity + importance * 0.3)
            
            # Adjust tempo based on event type
            if event.type in ["goal", "penalty"]:
                event_params.tempo = base_parameters.tempo + 10.0
            elif event.type in ["red_card", "yellow_card"]:
                event_params.tempo = base_parameters.tempo - 5.0
            
            # Create segment
            music_segment = MusicSegment(
                start_time=event_time,
                end_time=end_time,
                parameters=event_params,
                transition_in=True,
                transition_out=(i < len(relative_events) - 1),
                accent_time=event_time,
                accent_type=event.type
            )
            
            music_segments.append(music_segment)
        
        return music_segments