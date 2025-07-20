"""Music trigger engine module.

This module provides the main MusicTrigger class that maps events and emotions
to musical responses and parameters for the music generation engine.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.data.user_preferences import UserPreferences
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_trigger.event_mapping import EventMusicMapper
from crowd_sentiment_music_generator.services.music_trigger.emotion_mapping import EmotionMusicMapper
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class MusicTrigger:
    """Maps events and emotions to musical responses.
    
    This class provides methods for processing match events and crowd emotions,
    and determining appropriate musical responses and parameters. It integrates
    both event-based and emotion-based mapping to create a comprehensive music
    trigger engine that can respond to both discrete events and continuous
    emotional states.
    """
    
    def __init__(
        self, 
        config: Optional[SystemConfig] = None,
        music_engine_callback: Optional[Callable[[MusicalParameters], None]] = None,
        accent_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the music trigger engine.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
            music_engine_callback: Callback function for sending musical parameters to the music engine
            accent_callback: Callback function for triggering immediate musical accents
        """
        self.config = config or SystemConfig()
        self.event_mapper = EventMusicMapper(self.config)
        self.emotion_mapper = EmotionMusicMapper(self.config)
        self.music_engine_callback = music_engine_callback
        self.accent_callback = accent_callback
        self.current_parameters: Optional[MusicalParameters] = None
        self.last_event_time = 0.0
        self.last_emotion_time = 0.0
        self.event_cooldown = 5.0  # Minimum seconds between event triggers
        self.emotion_cooldown = 2.0  # Minimum seconds between emotion updates
        self.user_preferences: Optional[UserPreferences] = None
        logger.info("Initialized MusicTrigger engine")
    
    def set_music_engine_callback(
        self, 
        callback: Callable[[MusicalParameters], None]
    ) -> None:
        """Set the callback function for sending musical parameters to the music engine.
        
        Args:
            callback: Callback function that accepts MusicalParameters
        """
        self.music_engine_callback = callback
        logger.debug("Set music engine callback")
    
    def set_accent_callback(
        self, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Set the callback function for triggering immediate musical accents.
        
        Args:
            callback: Callback function that accepts a trigger dictionary
        """
        self.accent_callback = callback
        logger.debug("Set accent callback")
    
    def set_base_musical_state(self, tempo: float, key: str) -> None:
        """Set the base musical state for the trigger engine.
        
        Args:
            tempo: Base tempo in BPM
            key: Base musical key
        """
        self.event_mapper.set_base_musical_state(tempo, key)
        self.emotion_mapper.set_base_musical_state(tempo, key)
        
        # Initialize current parameters if not set
        if not self.current_parameters:
            self.current_parameters = MusicalParameters(
                tempo=tempo,
                key=key,
                intensity=0.5,
                instrumentation=["strings", "piano"],
                mood="neutral"
            )
        
        logger.debug(f"Set base musical state: tempo={tempo} BPM, key={key}")
    
    def set_user_preferences(self, preferences: UserPreferences) -> None:
        """Set user preferences for music generation.
        
        Args:
            preferences: User preferences
        """
        self.user_preferences = preferences
        
        # Set cultural style in emotion mapper
        if preferences.cultural_style:
            self.emotion_mapper.set_cultural_style(preferences.cultural_style)
        
        logger.debug(f"Set user preferences: {preferences}")
    
    @with_error_handling
    def process_event(
        self, 
        event: MatchEvent, 
        match_context: Optional[Dict[str, Any]] = None
    ) -> MusicalParameters:
        """Process a match event and determine musical response.
        
        This method maps the event to appropriate musical parameters and triggers
        immediate musical accents if needed.
        
        Args:
            event: Match event to process
            match_context: Optional match context information
            
        Returns:
            MusicalParameters object with parameters for the event
            
        Raises:
            MusicGenerationError: If event processing fails
        """
        # Check event cooldown to prevent too frequent triggers
        if event.timestamp - self.last_event_time < self.event_cooldown:
            logger.debug(f"Event {event.type} ignored due to cooldown")
            return self.current_parameters or MusicalParameters(
                tempo=100.0,
                key="C Major",
                intensity=0.5,
                instrumentation=["strings", "piano"],
                mood="neutral"
            )
        
        # Update last event time
        self.last_event_time = event.timestamp
        
        # Map event to musical parameters
        parameters = self.event_mapper.map_event_to_parameters(
            event, self.current_parameters
        )
        
        # Enrich parameters with match context if available
        if match_context:
            parameters = self.event_mapper.enrich_parameters_with_context(
                parameters, match_context
            )
        
        # Update current parameters
        self.current_parameters = parameters
        
        # Trigger immediate musical accent if callback is set
        if self.accent_callback:
            trigger = self.event_mapper.get_immediate_trigger(event)
            self.accent_callback(trigger)
        
        # Send parameters to music engine if callback is set
        if self.music_engine_callback:
            self.music_engine_callback(parameters)
        
        logger.info(f"Processed event {event.type} at {event.timestamp}")
        return parameters
    
    @with_error_handling
    def get_musical_parameters(
        self, 
        event_type: str, 
        emotion: str, 
        intensity: float
    ) -> Dict[str, Any]:
        """Get musical parameters based on event type, emotion and intensity.
        
        This method provides a simplified interface for getting musical parameters
        without creating full MatchEvent or CrowdEmotion objects.
        
        Args:
            event_type: Type of match event
            emotion: Crowd emotion
            intensity: Emotion intensity (0-100)
            
        Returns:
            Dictionary with musical parameters
            
        Raises:
            MusicGenerationError: If parameter retrieval fails
        """
        # Create a temporary event
        event = MatchEvent(
            id="temp",
            type=event_type,
            timestamp=0.0,
            team_id="temp"
        )
        
        # Map event to musical parameters
        parameters = self.event_mapper.map_event_to_parameters(
            event, self.current_parameters
        )
        
        # Adjust parameters based on emotion and intensity
        parameters.intensity = intensity / 100.0  # Convert to 0-1 scale
        
        # Convert to dictionary for easier consumption
        params_dict = parameters.dict()
        
        logger.debug(f"Generated musical parameters for {event_type}, {emotion}, {intensity}")
        return params_dict
    
    @with_error_handling
    def process_emotion(
        self, 
        emotion: CrowdEmotion, 
        match_context: Optional[Dict[str, Any]] = None
    ) -> MusicalParameters:
        """Process a crowd emotion and determine musical response.
        
        This method maps the emotion to appropriate musical parameters based on
        the emotion type, intensity, and current musical state.
        
        Args:
            emotion: Crowd emotion to process
            match_context: Optional match context information
            
        Returns:
            MusicalParameters object with parameters for the emotion
            
        Raises:
            MusicGenerationError: If emotion processing fails
        """
        # Check emotion cooldown to prevent too frequent updates
        if emotion.timestamp - self.last_emotion_time < self.emotion_cooldown:
            logger.debug(f"Emotion {emotion.emotion} ignored due to cooldown")
            return self.current_parameters or MusicalParameters(
                tempo=100.0,
                key="C Major",
                intensity=0.5,
                instrumentation=["strings", "piano"],
                mood="neutral"
            )
        
        # Update last emotion time
        self.last_emotion_time = emotion.timestamp
        
        # Map emotion to musical parameters
        parameters = self.emotion_mapper.map_emotion_to_parameters(
            emotion, self.current_parameters
        )
        
        # Apply user preferences if available
        if self.user_preferences:
            parameters = self.emotion_mapper.apply_user_preferences(
                parameters, self.user_preferences
            )
        
        # Update current parameters
        self.current_parameters = parameters
        
        # Send parameters to music engine if callback is set
        if self.music_engine_callback:
            self.music_engine_callback(parameters)
        
        logger.info(f"Processed emotion {emotion.emotion} (intensity: {emotion.intensity}) at {emotion.timestamp}")
        return parameters
    
    @with_error_handling
    def blend_event_and_emotion(
        self, 
        event: MatchEvent, 
        emotion: CrowdEmotion, 
        match_context: Optional[Dict[str, Any]] = None
    ) -> MusicalParameters:
        """Blend event-based and emotion-based musical parameters.
        
        This method creates a balanced blend of parameters from both event and emotion
        mapping, allowing for more nuanced musical responses that consider both
        discrete events and continuous emotional states.
        
        Args:
            event: Match event to process
            emotion: Crowd emotion to process
            match_context: Optional match context information
            
        Returns:
            Blended MusicalParameters object
            
        Raises:
            MusicGenerationError: If parameter blending fails
        """
        # Get parameters from event mapping
        event_parameters = self.event_mapper.map_event_to_parameters(
            event, self.current_parameters
        )
        
        # Get parameters from emotion mapping
        emotion_parameters = self.emotion_mapper.map_emotion_to_parameters(
            emotion, self.current_parameters
        )
        
        # Create a blend of the two parameter sets
        # For significant events, favor event parameters
        # For strong emotions, favor emotion parameters
        event_response = self.event_mapper.get_event_response(event.type)
        event_significance = event_response["intensity"]
        emotion_significance = emotion.intensity / 100.0 * emotion.confidence
        
        # Calculate blend weights
        total_significance = event_significance + emotion_significance
        event_weight = event_significance / total_significance if total_significance > 0 else 0.5
        emotion_weight = 1.0 - event_weight
        
        # Create blended parameters
        from copy import deepcopy
        blended = deepcopy(event_parameters)
        
        # Blend tempo
        blended.tempo = (event_parameters.tempo * event_weight + 
                         emotion_parameters.tempo * emotion_weight)
        
        # Blend intensity
        blended.intensity = (event_parameters.intensity * event_weight + 
                            emotion_parameters.intensity * emotion_weight)
        
        # For key, use event key for significant events, otherwise emotion key
        if event_significance > 0.7:
            blended.key = event_parameters.key
        else:
            blended.key = emotion_parameters.key
        
        # Blend instrumentation (combine both sets)
        blended.instrumentation = list(set(event_parameters.instrumentation + 
                                         emotion_parameters.instrumentation))
        
        # For mood, use event mood for significant events, otherwise emotion mood
        if event_significance > 0.7:
            blended.mood = event_parameters.mood
        else:
            blended.mood = emotion_parameters.mood
        
        # Blend transition duration
        if event_parameters.transition_duration and emotion_parameters.transition_duration:
            blended.transition_duration = (event_parameters.transition_duration * event_weight + 
                                         emotion_parameters.transition_duration * emotion_weight)
        
        # Apply user preferences if available
        if self.user_preferences:
            blended = self.emotion_mapper.apply_user_preferences(
                blended, self.user_preferences
            )
        
        # Update current parameters
        self.current_parameters = blended
        
        # Send parameters to music engine if callback is set
        if self.music_engine_callback:
            self.music_engine_callback(blended)
        
        logger.info(f"Blended event {event.type} and emotion {emotion.emotion}")
        return blended
    
    @with_error_handling
    def process_significant_events(
        self, 
        events: List[MatchEvent], 
        match_context: Optional[Dict[str, Any]] = None
    ) -> List[MusicalParameters]:
        """Process multiple events and determine the most significant musical responses.
        
        This method filters and prioritizes events to avoid overwhelming the music engine
        with too many parameter changes.
        
        Args:
            events: List of match events to process
            match_context: Optional match context information
            
        Returns:
            List of MusicalParameters objects for significant events
            
        Raises:
            MusicGenerationError: If event processing fails
        """
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Filter for significant events
        significant_events = []
        last_significant_time = 0.0
        
        for event in sorted_events:
            # Check if event is significant enough to trigger a musical change
            response = self.event_mapper.get_event_response(event.type)
            
            # Events with higher intensity are more significant
            if (response["intensity"] > 0.4 and 
                event.timestamp - last_significant_time > self.event_cooldown):
                significant_events.append(event)
                last_significant_time = event.timestamp
        
        # Process each significant event
        parameters_list = []
        for event in significant_events:
            try:
                parameters = self.process_event(event, match_context)
                parameters_list.append(parameters)
            except MusicGenerationError as e:
                logger.warning(f"Failed to process event {event.type}: {str(e)}")
                # Continue with next event
        
        return parameters_list
    
    @with_error_handling
    def scale_intensity(
        self, 
        intensity_level: float
    ) -> MusicalParameters:
        """Scale current musical parameters based on a new intensity level.
        
        This method provides a simple way to adjust the intensity of the current
        musical state without changing other parameters.
        
        Args:
            intensity_level: New intensity level (0-100)
            
        Returns:
            Scaled MusicalParameters object
            
        Raises:
            MusicGenerationError: If parameter scaling fails
        """
        if not self.current_parameters:
            # Create default parameters if none exist
            self.current_parameters = MusicalParameters(
                tempo=100.0,
                key="C Major",
                intensity=0.5,
                instrumentation=["strings", "piano"],
                mood="neutral"
            )
        
        # Scale parameters with intensity
        scaled_parameters = self.emotion_mapper.scale_parameters_with_intensity(
            self.current_parameters, intensity_level
        )
        
        # Apply user preferences if available
        if self.user_preferences:
            scaled_parameters = self.emotion_mapper.apply_user_preferences(
                scaled_parameters, self.user_preferences
            )
        
        # Update current parameters
        self.current_parameters = scaled_parameters
        
        # Send parameters to music engine if callback is set
        if self.music_engine_callback:
            self.music_engine_callback(scaled_parameters)
        
        logger.info(f"Scaled parameters to intensity level {intensity_level}")
        return scaled_parameters