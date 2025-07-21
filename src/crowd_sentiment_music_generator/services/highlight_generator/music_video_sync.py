"""Music-video synchronization module for aligning music with video segments."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic, MusicSegment
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class SyncPoint(BaseModel):
    """Model for a synchronization point between music and video.
    
    Attributes:
        video_time: Time in the video (seconds)
        music_time: Time in the music (seconds)
        importance: Importance of this sync point (0-1)
        type: Type of sync point (e.g., "event", "transition", "beat")
    """
    
    video_time: float
    music_time: float
    importance: float = 1.0
    type: str = "event"


class MusicVideoSynchronizer:
    """Synchronizes music with video segments.
    
    This class provides methods for aligning music with video segments,
    adjusting durations, and generating transitions between segments.
    
    Attributes:
        music_engine: Magenta music engine for generating music
        logger: Logger instance
    """
    
    def __init__(self, music_engine: Optional[MagentaMusicEngine] = None):
        """Initialize music-video synchronizer.
        
        Args:
            music_engine: Optional Magenta music engine for generating music
        """
        self.music_engine = music_engine
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    def align_music_to_video(
        self, highlight_music: HighlightMusic, segment: HighlightSegment
    ) -> HighlightMusic:
        """Align music composition to video segment.
        
        Args:
            highlight_music: Music composition to align
            segment: Video segment to align with
            
        Returns:
            Aligned music composition
            
        Raises:
            ValueError: If highlight_music and segment IDs don't match
        """
        # Verify that highlight_music and segment match
        if highlight_music.highlight_id != segment.id:
            raise ValueError(
                f"Highlight music ID ({highlight_music.highlight_id}) "
                f"doesn't match segment ID ({segment.id})"
            )
        
        # Create sync points
        sync_points = self._create_sync_points(highlight_music, segment)
        
        # Align music segments to sync points
        aligned_segments = self._align_segments_to_sync_points(
            highlight_music.segments, sync_points
        )
        
        # Create aligned highlight music
        aligned_music = HighlightMusic(
            highlight_id=segment.id,
            segments=aligned_segments,
            base_parameters=highlight_music.base_parameters,
            duration=segment.duration,
            metadata=highlight_music.metadata
        )
        
        self.logger.info(
            f"Aligned music to video for highlight segment {segment.id} "
            f"with {len(sync_points)} sync points"
        )
        
        return aligned_music
    
    @with_error_handling
    def adjust_duration(
        self, highlight_music: HighlightMusic, target_duration: float
    ) -> HighlightMusic:
        """Adjust music composition to match target duration.
        
        This method uses a smart scaling approach that preserves important segments
        while adjusting less important ones more aggressively.
        
        Args:
            highlight_music: Music composition to adjust
            target_duration: Target duration in seconds
            
        Returns:
            Adjusted music composition
            
        Raises:
            ValueError: If target_duration is not positive
        """
        if target_duration <= 0:
            raise ValueError("Target duration must be positive")
        
        # Calculate duration ratio
        original_duration = highlight_music.duration
        ratio = target_duration / original_duration
        
        # If ratio is close to 1, use simple scaling
        if 0.9 <= ratio <= 1.1:
            return self._adjust_duration_simple(highlight_music, target_duration)
        
        # For more significant adjustments, use smart scaling
        return self._adjust_duration_smart(highlight_music, target_duration)
    
    def _adjust_duration_simple(
        self, highlight_music: HighlightMusic, target_duration: float
    ) -> HighlightMusic:
        """Simple proportional scaling of all segments.
        
        Args:
            highlight_music: Music composition to adjust
            target_duration: Target duration in seconds
            
        Returns:
            Adjusted music composition
        """
        original_duration = highlight_music.duration
        ratio = target_duration / original_duration
        
        # Adjust segment durations
        adjusted_segments = []
        current_time = 0.0
        
        for segment in highlight_music.segments:
            # Calculate new duration
            original_duration = segment.end_time - segment.start_time
            new_duration = original_duration * ratio
            
            # Create adjusted segment
            adjusted_segment = MusicSegment(
                start_time=current_time,
                end_time=current_time + new_duration,
                parameters=segment.parameters,
                transition_in=segment.transition_in,
                transition_out=segment.transition_out,
                accent_time=(segment.accent_time - segment.start_time) * ratio + current_time if segment.accent_time else None,
                accent_type=segment.accent_type
            )
            
            adjusted_segments.append(adjusted_segment)
            current_time += new_duration
        
        # Create adjusted highlight music
        adjusted_music = HighlightMusic(
            highlight_id=highlight_music.highlight_id,
            segments=adjusted_segments,
            base_parameters=highlight_music.base_parameters,
            duration=target_duration,
            metadata=highlight_music.metadata
        )
        
        self.logger.info(
            f"Simple duration adjustment from {original_duration:.2f}s to {target_duration:.2f}s "
            f"for highlight {highlight_music.highlight_id}"
        )
        
        return adjusted_music
    
    def _adjust_duration_smart(
        self, highlight_music: HighlightMusic, target_duration: float
    ) -> HighlightMusic:
        """Smart scaling that preserves important segments.
        
        Args:
            highlight_music: Music composition to adjust
            target_duration: Target duration in seconds
            
        Returns:
            Adjusted music composition
        """
        original_duration = highlight_music.duration
        
        # Calculate segment importance scores
        segments_with_scores = []
        for segment in highlight_music.segments:
            # Base importance on segment parameters
            importance = self._calculate_segment_importance(segment)
            segments_with_scores.append((segment, importance))
        
        # Normalize importance scores
        total_importance = sum(score for _, score in segments_with_scores)
        if total_importance > 0:
            segments_with_scores = [(segment, score / total_importance) 
                                   for segment, score in segments_with_scores]
        
        # Calculate target durations for each segment
        # More important segments get scaled less, less important segments get scaled more
        total_adjustment = target_duration - original_duration
        
        # Calculate weighted adjustments
        weighted_adjustments = []
        for segment, importance in segments_with_scores:
            segment_duration = segment.end_time - segment.start_time
            
            # Inverse relationship: higher importance = less adjustment
            # We use (1 - sqrt(importance)) to make the relationship non-linear
            # This ensures very important segments are preserved much more than less important ones
            adjustment_weight = 1.0 - (importance ** 0.5)
            
            # Ensure minimum adjustment weight
            adjustment_weight = max(0.1, adjustment_weight)
            
            weighted_adjustments.append((segment, segment_duration, adjustment_weight))
        
        # Normalize adjustment weights
        total_weight = sum(weight for _, _, weight in weighted_adjustments)
        if total_weight > 0:
            weighted_adjustments = [(segment, duration, weight / total_weight) 
                                   for segment, duration, weight in weighted_adjustments]
        
        # Calculate new durations
        new_segments = []
        current_time = 0.0
        
        for segment, original_duration, weight in weighted_adjustments:
            # Calculate adjustment for this segment
            segment_adjustment = total_adjustment * weight
            
            # Calculate new duration
            new_duration = max(0.5, original_duration + segment_adjustment)
            
            # Ensure minimum segment duration (0.5 seconds)
            if new_duration < 0.5:
                new_duration = 0.5
            
            # Create adjusted segment
            adjusted_segment = MusicSegment(
                start_time=current_time,
                end_time=current_time + new_duration,
                parameters=segment.parameters,
                transition_in=segment.transition_in,
                transition_out=segment.transition_out,
                accent_time=(segment.accent_time - segment.start_time) * (new_duration / original_duration) + current_time 
                    if segment.accent_time is not None else None,
                accent_type=segment.accent_type
            )
            
            new_segments.append(adjusted_segment)
            current_time += new_duration
        
        # Final adjustment to ensure exact target duration
        if new_segments:
            duration_diff = target_duration - current_time
            last_segment = new_segments[-1]
            new_segments[-1] = MusicSegment(
                start_time=last_segment.start_time,
                end_time=last_segment.end_time + duration_diff,
                parameters=last_segment.parameters,
                transition_in=last_segment.transition_in,
                transition_out=last_segment.transition_out,
                accent_time=last_segment.accent_time,
                accent_type=last_segment.accent_type
            )
        
        # Create adjusted highlight music
        adjusted_music = HighlightMusic(
            highlight_id=highlight_music.highlight_id,
            segments=new_segments,
            base_parameters=highlight_music.base_parameters,
            duration=target_duration,
            metadata=highlight_music.metadata
        )
        
        self.logger.info(
            f"Smart duration adjustment from {original_duration:.2f}s to {target_duration:.2f}s "
            f"for highlight {highlight_music.highlight_id}"
        )
        
        return adjusted_music
    
    def _calculate_segment_importance(self, segment: MusicSegment) -> float:
        """Calculate importance score for a segment.
        
        Args:
            segment: Music segment
            
        Returns:
            Importance score (0-1)
        """
        # Base importance
        importance = 0.5
        
        # Segments with accents are more important
        if segment.accent_time is not None:
            importance += 0.3
            
            # Certain accent types are more important
            if segment.accent_type in ["goal", "penalty", "red_card"]:
                importance += 0.1
        
        # High intensity segments are more important
        importance += segment.parameters.intensity * 0.2
        
        # Cap at 1.0
        return min(1.0, importance)
    
    @with_error_handling
    def generate_transitions(self, highlight_music: HighlightMusic) -> HighlightMusic:
        """Generate smooth transitions between music segments.
        
        This method creates appropriate transitions between music segments based on
        the musical contrast between them. Higher contrast segments get longer,
        more sophisticated transitions.
        
        Args:
            highlight_music: Music composition to add transitions to
            
        Returns:
            Music composition with transitions
            
        Raises:
            MusicGenerationError: If transition generation fails
        """
        # If there's only one segment, no transitions needed
        if len(highlight_music.segments) <= 1:
            return highlight_music
        
        # Process each segment
        segments_with_transitions = []
        
        for i, segment in enumerate(highlight_music.segments):
            # Add current segment
            segments_with_transitions.append(segment)
            
            # If not the last segment and transition_out is True
            if i < len(highlight_music.segments) - 1 and segment.transition_out:
                next_segment = highlight_music.segments[i + 1]
                
                # Only add transition if next segment has transition_in
                if next_segment.transition_in:
                    # Calculate musical contrast between segments
                    contrast = self._calculate_musical_contrast(segment, next_segment)
                    
                    # Create transition segment based on contrast level
                    if contrast > 0.7:  # High contrast
                        transition = self._create_complex_transition(segment, next_segment)
                    elif contrast > 0.3:  # Medium contrast
                        transition = self._create_standard_transition(segment, next_segment)
                    else:  # Low contrast
                        transition = self._create_simple_transition(segment, next_segment)
                    
                    segments_with_transitions.append(transition)
        
        # Recalculate segment times to ensure proper timing
        segments_with_transitions = self._recalculate_segment_times(segments_with_transitions)
        
        # Create highlight music with transitions
        music_with_transitions = HighlightMusic(
            highlight_id=highlight_music.highlight_id,
            segments=segments_with_transitions,
            base_parameters=highlight_music.base_parameters,
            duration=highlight_music.duration,
            metadata=highlight_music.metadata
        )
        
        self.logger.info(
            f"Generated transitions for highlight {highlight_music.highlight_id} "
            f"(segments: {len(highlight_music.segments)} → {len(segments_with_transitions)})"
        )
        
        return music_with_transitions
    
    def _calculate_musical_contrast(
        self, segment1: MusicSegment, segment2: MusicSegment
    ) -> float:
        """Calculate musical contrast between two segments.
        
        Args:
            segment1: First segment
            segment2: Second segment
            
        Returns:
            Contrast score (0-1)
        """
        # Calculate contrast based on musical parameters
        params1 = segment1.parameters
        params2 = segment2.parameters
        
        # Tempo contrast (normalized to 0-1 range)
        tempo_diff = abs(params1.tempo - params2.tempo) / 100.0
        tempo_contrast = min(1.0, tempo_diff)
        
        # Intensity contrast
        intensity_contrast = abs(params1.intensity - params2.intensity)
        
        # Key contrast (binary: same or different)
        key_contrast = 0.0 if params1.key == params2.key else 0.8
        
        # Mood contrast (binary: same or different)
        mood_contrast = 0.0 if params1.mood == params2.mood else 0.7
        
        # Instrumentation contrast (based on overlap)
        common_instruments = set(params1.instrumentation).intersection(set(params2.instrumentation))
        total_instruments = set(params1.instrumentation).union(set(params2.instrumentation))
        
        instrumentation_contrast = 0.0
        if total_instruments:
            instrumentation_contrast = 1.0 - (len(common_instruments) / len(total_instruments))
        
        # Accent type contrast (if both have accents)
        accent_contrast = 0.0
        if segment1.accent_type and segment2.accent_type:
            accent_contrast = 0.0 if segment1.accent_type == segment2.accent_type else 0.5
        
        # Weighted combination of all contrast factors
        contrast = (
            tempo_contrast * 0.25 +
            intensity_contrast * 0.3 +
            key_contrast * 0.2 +
            mood_contrast * 0.15 +
            instrumentation_contrast * 0.05 +
            accent_contrast * 0.05
        )
        
        return min(1.0, contrast)
    
    def _create_simple_transition(
        self, from_segment: MusicSegment, to_segment: MusicSegment
    ) -> MusicSegment:
        """Create a simple transition for low contrast segments.
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            
        Returns:
            Transition segment
        """
        # Short transition duration
        transition_duration = 0.5
        
        # Use transition_duration from parameters if available
        if from_segment.parameters.transition_duration is not None:
            transition_duration = min(from_segment.parameters.transition_duration, 1.0)
        elif to_segment.parameters.transition_duration is not None:
            transition_duration = min(to_segment.parameters.transition_duration, 1.0)
        
        # Ensure transition isn't longer than 20% of either segment
        from_duration = from_segment.end_time - from_segment.start_time
        to_duration = to_segment.end_time - to_segment.start_time
        max_transition = min(from_duration, to_duration) * 0.2
        transition_duration = min(transition_duration, max_transition)
        
        # Calculate start and end times
        start_time = to_segment.start_time - transition_duration
        end_time = to_segment.start_time
        
        # For simple transitions, favor the target segment's parameters
        transition_params = to_segment.parameters.copy()
        
        # Create transition segment
        transition_segment = MusicSegment(
            start_time=start_time,
            end_time=end_time,
            parameters=transition_params,
            transition_in=False,
            transition_out=False
        )
        
        return transition_segment
    
    def _create_standard_transition(
        self, from_segment: MusicSegment, to_segment: MusicSegment
    ) -> MusicSegment:
        """Create a standard transition for medium contrast segments.
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            
        Returns:
            Transition segment
        """
        # Medium transition duration
        transition_duration = 1.5
        
        # Use transition_duration from parameters if available
        if from_segment.parameters.transition_duration is not None:
            transition_duration = from_segment.parameters.transition_duration
        elif to_segment.parameters.transition_duration is not None:
            transition_duration = to_segment.parameters.transition_duration
        
        # Ensure transition isn't longer than 30% of either segment
        from_duration = from_segment.end_time - from_segment.start_time
        to_duration = to_segment.end_time - to_segment.start_time
        max_transition = min(from_duration, to_duration) * 0.3
        transition_duration = min(transition_duration, max_transition)
        
        # Calculate start and end times
        start_time = to_segment.start_time - transition_duration
        end_time = to_segment.start_time
        
        # Interpolate parameters
        transition_params = self._interpolate_parameters(
            from_segment.parameters, to_segment.parameters
        )
        
        # Create transition segment
        transition_segment = MusicSegment(
            start_time=start_time,
            end_time=end_time,
            parameters=transition_params,
            transition_in=False,
            transition_out=False
        )
        
        return transition_segment
    
    def _create_complex_transition(
        self, from_segment: MusicSegment, to_segment: MusicSegment
    ) -> MusicSegment:
        """Create a complex transition for high contrast segments.
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            
        Returns:
            Transition segment
        """
        # Longer transition duration
        transition_duration = 2.5
        
        # Use transition_duration from parameters if available
        if from_segment.parameters.transition_duration is not None:
            transition_duration = max(from_segment.parameters.transition_duration, 2.0)
        elif to_segment.parameters.transition_duration is not None:
            transition_duration = max(to_segment.parameters.transition_duration, 2.0)
        
        # Ensure transition isn't longer than 40% of either segment
        from_duration = from_segment.end_time - from_segment.start_time
        to_duration = to_segment.end_time - to_segment.start_time
        max_transition = min(from_duration, to_duration) * 0.4
        transition_duration = min(transition_duration, max_transition)
        
        # Calculate start and end times
        start_time = to_segment.start_time - transition_duration
        end_time = to_segment.start_time
        
        # For complex transitions, create a more sophisticated parameter blend
        # that emphasizes the contrast between segments
        
        # Start with base parameters from source segment
        transition_params = from_segment.parameters.copy()
        
        # Gradually shift tempo toward target
        transition_params.tempo = from_segment.parameters.tempo * 0.7 + to_segment.parameters.tempo * 0.3
        
        # For key, use a transitional key if different
        if from_segment.parameters.key != to_segment.parameters.key:
            # In a real implementation, this would use music theory to find a transitional key
            # For now, we'll just use the source key
            transition_params.key = from_segment.parameters.key
        
        # For mood, create a transitional mood
        if from_segment.parameters.mood != to_segment.parameters.mood:
            # Use a neutral mood for transition
            transition_params.mood = "transitional"
        
        # For instrumentation, use a subset that works well for transitions
        transition_instruments = ["strings", "synth_pad"]
        if "piano" in from_segment.parameters.instrumentation or "piano" in to_segment.parameters.instrumentation:
            transition_instruments.append("piano")
        
        transition_params.instrumentation = transition_instruments
        
        # Set a specific transition duration
        transition_params.transition_duration = transition_duration
        
        # Create transition segment
        transition_segment = MusicSegment(
            start_time=start_time,
            end_time=end_time,
            parameters=transition_params,
            transition_in=False,
            transition_out=False
        )
        
        return transition_segment
    
    def _recalculate_segment_times(self, segments: List[MusicSegment]) -> List[MusicSegment]:
        """Recalculate segment times to ensure proper timing.
        
        Args:
            segments: List of music segments
            
        Returns:
            List of segments with recalculated times
        """
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Recalculate times to ensure no overlaps
        recalculated_segments = []
        current_time = 0.0
        
        for segment in sorted_segments:
            # Calculate segment duration
            duration = segment.end_time - segment.start_time
            
            # Create new segment with recalculated times
            new_segment = MusicSegment(
                start_time=current_time,
                end_time=current_time + duration,
                parameters=segment.parameters,
                transition_in=segment.transition_in,
                transition_out=segment.transition_out,
                accent_time=(segment.accent_time - segment.start_time) + current_time if segment.accent_time is not None else None,
                accent_type=segment.accent_type
            )
            
            recalculated_segments.append(new_segment)
            current_time += duration
        
        return recalculated_segments
    
    def _create_sync_points(
        self, highlight_music: HighlightMusic, segment: HighlightSegment
    ) -> List[SyncPoint]:
        """Create synchronization points between music and video.
        
        Args:
            highlight_music: Music composition
            segment: Video segment
            
        Returns:
            List of synchronization points
        """
        sync_points = []
        
        # Add start point
        sync_points.append(
            SyncPoint(
                video_time=0.0,
                music_time=0.0,
                importance=1.0,
                type="start"
            )
        )
        
        # Add end point
        sync_points.append(
            SyncPoint(
                video_time=segment.duration,
                music_time=highlight_music.duration,
                importance=1.0,
                type="end"
            )
        )
        
        # Add key moment point
        key_moment_relative = segment.key_moment_time - segment.start_time
        
        # Find closest music segment for key moment
        closest_segment = None
        min_distance = float('inf')
        
        # First priority: segments with accent_time
        for music_segment in highlight_music.segments:
            if music_segment.accent_time is not None:
                distance = abs(music_segment.accent_time - key_moment_relative)
                if distance < min_distance:
                    min_distance = distance
                    closest_segment = music_segment
        
        # If no segment with accent_time found, look for segments with high intensity
        if closest_segment is None:
            for music_segment in highlight_music.segments:
                # Check if segment contains the key moment time
                if music_segment.start_time <= key_moment_relative <= music_segment.end_time:
                    segment_midpoint = (music_segment.start_time + music_segment.end_time) / 2
                    distance = abs(segment_midpoint - key_moment_relative)
                    
                    # Weight distance by intensity (higher intensity = higher priority)
                    weighted_distance = distance / (music_segment.parameters.intensity + 0.1)
                    
                    if weighted_distance < min_distance:
                        min_distance = weighted_distance
                        closest_segment = music_segment
        
        # If found a segment, add sync point
        if closest_segment:
            # If segment has accent_time, use it
            if closest_segment.accent_time is not None:
                sync_points.append(
                    SyncPoint(
                        video_time=key_moment_relative,
                        music_time=closest_segment.accent_time,
                        importance=0.9,
                        type="key_moment"
                    )
                )
            else:
                # Otherwise, use segment midpoint or calculate optimal point based on intensity curve
                segment_duration = closest_segment.end_time - closest_segment.start_time
                
                # For segments with high intensity, use peak point (2/3 through segment)
                # For segments with low intensity, use earlier point (1/3 through segment)
                intensity_factor = closest_segment.parameters.intensity
                relative_position = 0.33 + (intensity_factor * 0.33)  # Range: 0.33-0.66
                
                optimal_time = closest_segment.start_time + (segment_duration * relative_position)
                
                sync_points.append(
                    SyncPoint(
                        video_time=key_moment_relative,
                        music_time=optimal_time,
                        importance=0.85,
                        type="key_moment"
                    )
                )
        
        # Add segment boundary points with varying importance based on musical parameters
        for i, music_segment in enumerate(highlight_music.segments):
            if i > 0:  # Skip first segment start
                # Calculate importance based on musical contrast
                prev_segment = highlight_music.segments[i-1]
                
                # Calculate musical contrast between segments
                tempo_diff = abs(music_segment.parameters.tempo - prev_segment.parameters.tempo) / 100.0
                intensity_diff = abs(music_segment.parameters.intensity - prev_segment.parameters.intensity)
                mood_contrast = 0.2 if music_segment.parameters.mood != prev_segment.parameters.mood else 0.0
                
                # Combined contrast score (0.0-1.0)
                contrast = min(1.0, (tempo_diff + intensity_diff + mood_contrast) / 2.0)
                
                # Higher contrast = higher importance for sync point
                importance = 0.4 + (contrast * 0.4)  # Range: 0.4-0.8
                
                sync_points.append(
                    SyncPoint(
                        video_time=music_segment.start_time,
                        music_time=music_segment.start_time,
                        importance=importance,
                        type="segment_boundary"
                    )
                )
        
        # Add additional sync points for significant musical events
        for music_segment in highlight_music.segments:
            # If segment has high intensity, add additional sync points
            if music_segment.parameters.intensity > 0.8:
                segment_duration = music_segment.end_time - music_segment.start_time
                
                # Only add for longer segments
                if segment_duration > 5.0:
                    # Add sync point at intensity peak (typically 2/3 through high-intensity segment)
                    peak_time = music_segment.start_time + (segment_duration * 0.66)
                    
                    # Only add if not too close to existing sync points
                    if not any(abs(sp.music_time - peak_time) < 1.0 for sp in sync_points):
                        sync_points.append(
                            SyncPoint(
                                video_time=peak_time,
                                music_time=peak_time,
                                importance=0.7,
                                type="intensity_peak"
                            )
                        )
        
        # Sort sync points by music_time to ensure proper ordering
        sync_points.sort(key=lambda sp: sp.music_time)
        
        self.logger.info(
            f"Created {len(sync_points)} sync points for highlight {segment.id}: "
            f"{', '.join(sp.type for sp in sync_points)}"
        )
        
        return sync_points
    
    def _align_segments_to_sync_points(
        self, segments: List[MusicSegment], sync_points: List[SyncPoint]
    ) -> List[MusicSegment]:
        """Align music segments to sync points.
        
        Args:
            segments: List of music segments
            sync_points: List of synchronization points
            
        Returns:
            List of aligned music segments
        """
        # Sort sync points by video time
        sorted_sync_points = sorted(sync_points, key=lambda sp: sp.video_time)
        
        # Create mapping function from original music time to aligned time
        def map_time(original_time: float) -> float:
            # Find surrounding sync points
            prev_point = sorted_sync_points[0]
            next_point = sorted_sync_points[-1]
            
            for i, point in enumerate(sorted_sync_points):
                if point.music_time > original_time:
                    next_point = point
                    prev_point = sorted_sync_points[i - 1] if i > 0 else point
                    break
                elif point.music_time == original_time:
                    return point.video_time
            
            # Linear interpolation between sync points
            if prev_point.music_time == next_point.music_time:
                return prev_point.video_time
            
            ratio = ((original_time - prev_point.music_time) / 
                     (next_point.music_time - prev_point.music_time))
            
            return prev_point.video_time + ratio * (next_point.video_time - prev_point.video_time)
        
        # Align segments
        aligned_segments = []
        
        for segment in segments:
            # Map start and end times
            aligned_start = map_time(segment.start_time)
            aligned_end = map_time(segment.end_time)
            
            # Map accent time if present
            aligned_accent = map_time(segment.accent_time) if segment.accent_time is not None else None
            
            # Create aligned segment
            aligned_segment = MusicSegment(
                start_time=aligned_start,
                end_time=aligned_end,
                parameters=segment.parameters,
                transition_in=segment.transition_in,
                transition_out=segment.transition_out,
                accent_time=aligned_accent,
                accent_type=segment.accent_type
            )
            
            aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def _create_transition_segment(
        self, from_segment: MusicSegment, to_segment: MusicSegment
    ) -> MusicSegment:
        """Create a transition segment between two music segments.
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            
        Returns:
            Transition segment
        """
        # Default transition duration (in seconds)
        transition_duration = 1.0
        
        # Use transition_duration from parameters if available
        if from_segment.parameters.transition_duration is not None:
            transition_duration = from_segment.parameters.transition_duration
        elif to_segment.parameters.transition_duration is not None:
            transition_duration = to_segment.parameters.transition_duration
        
        # Ensure transition isn't longer than half of either segment
        from_duration = from_segment.end_time - from_segment.start_time
        to_duration = to_segment.end_time - to_segment.start_time
        max_transition = min(from_duration, to_duration) / 2
        transition_duration = min(transition_duration, max_transition)
        
        # Calculate start and end times
        start_time = to_segment.start_time - transition_duration
        end_time = to_segment.start_time
        
        # Interpolate parameters
        transition_params = self._interpolate_parameters(
            from_segment.parameters, to_segment.parameters
        )
        
        # Create transition segment
        transition_segment = MusicSegment(
            start_time=start_time,
            end_time=end_time,
            parameters=transition_params,
            transition_in=False,
            transition_out=False
        )
        
        return transition_segment
    
    def _interpolate_parameters(
        self, from_params: MusicalParameters, to_params: MusicalParameters
    ) -> MusicalParameters:
        """Interpolate between two sets of musical parameters.
        
        Args:
            from_params: Source parameters
            to_params: Target parameters
            
        Returns:
            Interpolated parameters
        """
        # Create a copy of from_params
        params = from_params.copy()
        
        # Interpolate numeric parameters (at 50%)
        params.tempo = (from_params.tempo + to_params.tempo) / 2
        params.intensity = (from_params.intensity + to_params.intensity) / 2
        
        # For key, use from_params key
        # For mood, use from_params mood
        
        # For instrumentation, combine both sets
        params.instrumentation = list(set(from_params.instrumentation + to_params.instrumentation))
        
        # For transition duration, use average if both are defined
        if from_params.transition_duration is not None and to_params.transition_duration is not None:
            params.transition_duration = (from_params.transition_duration + to_params.transition_duration) / 2
        elif from_params.transition_duration is not None:
            params.transition_duration = from_params.transition_duration
        elif to_params.transition_duration is not None:
            params.transition_duration = to_params.transition_duration
        
        return params