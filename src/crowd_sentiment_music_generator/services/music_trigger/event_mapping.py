"""Event-to-music mapping module.

This module provides functionality for mapping match events to musical responses,
including configuration for different event types and parameter generation for musical responses.
"""

import logging
from typing import Dict, Any, Optional, List

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class EventMusicMapper:
    """Maps match events to musical responses.
    
    This class provides methods for mapping different types of match events
    to appropriate musical parameters and responses.
    """
    
    # Event musical responses configuration
    EVENT_MUSICAL_RESPONSES = {
        "goal": {
            "immediate": "cymbal_crash + brass_fanfare",
            "evolution": "major_key_celebration",
            "duration": 30,
            "intensity": 1.0,
            "tempo_change": 20,  # BPM increase
            "key_preference": "major",
            "instrumentation": ["brass", "percussion", "strings"],
            "mood": "triumphant"
        },
        "penalty_awarded": {
            "immediate": "tension_drums",
            "evolution": "building_suspense",
            "duration": 20,
            "intensity": 0.7,
            "tempo_change": -5,  # BPM decrease
            "key_preference": "minor",
            "instrumentation": ["percussion", "strings"],
            "mood": "suspenseful"
        },
        "near_miss": {
            "immediate": "string_rise",
            "evolution": "anticipation_release",
            "duration": 10,
            "intensity": 0.6,
            "tempo_change": 10,  # BPM increase
            "key_preference": "minor",
            "instrumentation": ["strings", "woodwinds"],
            "mood": "tense"
        },
        "red_card": {
            "immediate": "dramatic_hit",
            "evolution": "tension_build",
            "duration": 15,
            "intensity": 0.8,
            "tempo_change": -10,  # BPM decrease
            "key_preference": "diminished",
            "instrumentation": ["brass", "percussion"],
            "mood": "dramatic"
        },
        "yellow_card": {
            "immediate": "discord_accent",
            "evolution": "minor_tension",
            "duration": 15,
            "intensity": 0.5,
            "tempo_change": -5,  # BPM decrease
            "key_preference": "minor",
            "instrumentation": ["strings", "percussion"],
            "mood": "cautious"
        },
        "corner": {
            "immediate": "percussion_roll",
            "evolution": "anticipation_build",
            "duration": 10,
            "intensity": 0.4,
            "tempo_change": 5,  # BPM increase
            "key_preference": "neutral",
            "instrumentation": ["percussion", "strings"],
            "mood": "anticipatory"
        },
        "free_kick": {
            "immediate": "soft_percussion",
            "evolution": "anticipation_light",
            "duration": 10,
            "intensity": 0.4,
            "tempo_change": 0,  # No change
            "key_preference": "neutral",
            "instrumentation": ["strings", "woodwinds"],
            "mood": "focused"
        },
        "substitution": {
            "immediate": "transition_sweep",
            "evolution": "neutral_transition",
            "duration": 8,
            "intensity": 0.3,
            "tempo_change": 0,  # No change
            "key_preference": "neutral",
            "instrumentation": ["strings", "woodwinds"],
            "mood": "transitional"
        },
        "injury": {
            "immediate": "somber_tone",
            "evolution": "concern_build",
            "duration": 15,
            "intensity": 0.4,
            "tempo_change": -10,  # BPM decrease
            "key_preference": "minor",
            "instrumentation": ["strings", "woodwinds"],
            "mood": "concerned"
        },
        "offside": {
            "immediate": "short_interruption",
            "evolution": "brief_pause",
            "duration": 5,
            "intensity": 0.3,
            "tempo_change": -5,  # BPM decrease
            "key_preference": "neutral",
            "instrumentation": ["woodwinds"],
            "mood": "interrupted"
        },
        "kickoff": {
            "immediate": "opening_fanfare",
            "evolution": "anticipation_build",
            "duration": 20,
            "intensity": 0.6,
            "tempo_change": 5,  # BPM increase
            "key_preference": "major",
            "instrumentation": ["brass", "strings", "percussion"],
            "mood": "anticipatory"
        },
        "halftime": {
            "immediate": "transition_sweep",
            "evolution": "reflective_pause",
            "duration": 15,
            "intensity": 0.4,
            "tempo_change": -10,  # BPM decrease
            "key_preference": "neutral",
            "instrumentation": ["strings", "woodwinds"],
            "mood": "reflective"
        },
        "fulltime": {
            "immediate": "final_resolution",
            "evolution": "conclusion_theme",
            "duration": 30,
            "intensity": 0.7,
            "tempo_change": 0,  # No change
            "key_preference": "major",
            "instrumentation": ["full_orchestra"],
            "mood": "conclusive"
        }
    }
    
    # Default response for unknown event types
    DEFAULT_RESPONSE = {
        "immediate": "subtle_accent",
        "evolution": "neutral_continuation",
        "duration": 5,
        "intensity": 0.3,
        "tempo_change": 0,  # No change
        "key_preference": "neutral",
        "instrumentation": ["strings"],
        "mood": "neutral"
    }
    
    # Key mapping for different emotions and preferences
    KEY_MAPPING = {
        "major": ["C Major", "G Major", "D Major", "A Major", "E Major", "F Major"],
        "minor": ["A Minor", "E Minor", "D Minor", "G Minor", "C Minor", "F Minor"],
        "diminished": ["B Diminished", "D Diminished", "F# Diminished"],
        "neutral": ["C Major", "A Minor"]  # Default keys
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the event music mapper.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.base_tempo = 100.0  # Default base tempo in BPM
        self.current_key = "C Major"  # Default key
        logger.info("Initialized EventMusicMapper")
    
    def set_base_musical_state(self, tempo: float, key: str) -> None:
        """Set the base musical state for event mapping.
        
        Args:
            tempo: Base tempo in BPM
            key: Base musical key
        """
        self.base_tempo = tempo
        self.current_key = key
        logger.debug(f"Set base musical state: tempo={tempo} BPM, key={key}")
    
    @with_error_handling
    def get_event_response(self, event_type: str) -> Dict[str, Any]:
        """Get the musical response configuration for a specific event type.
        
        Args:
            event_type: Type of match event
            
        Returns:
            Dictionary with musical response configuration
            
        Raises:
            MusicGenerationError: If event response retrieval fails
        """
        # Get response configuration for the event type, or use default if not found
        response = self.EVENT_MUSICAL_RESPONSES.get(event_type, self.DEFAULT_RESPONSE)
        logger.debug(f"Retrieved musical response for event type: {event_type}")
        return response
    
    @with_error_handling
    def map_event_to_parameters(
        self, 
        event: MatchEvent, 
        current_parameters: Optional[MusicalParameters] = None
    ) -> MusicalParameters:
        """Map a match event to musical parameters.
        
        This method generates appropriate musical parameters based on the event type
        and current musical state.
        
        Args:
            event: Match event to map
            current_parameters: Current musical parameters (optional)
            
        Returns:
            MusicalParameters object with parameters for the event
            
        Raises:
            MusicGenerationError: If parameter generation fails
        """
        # Get response configuration for the event type
        response = self.get_event_response(event.type)
        
        # Use current parameters as base if provided, otherwise use defaults
        base_tempo = current_parameters.tempo if current_parameters else self.base_tempo
        base_key = current_parameters.key if current_parameters else self.current_key
        
        # Calculate new tempo based on event response
        new_tempo = base_tempo + response["tempo_change"]
        
        # Ensure tempo stays within reasonable bounds
        new_tempo = max(60.0, min(180.0, new_tempo))
        
        # Select key based on key preference
        key_preference = response["key_preference"]
        if key_preference != "neutral" or not current_parameters:
            # Change key if preference is specified or no current parameters
            new_key = self._select_key(key_preference)
        else:
            # Keep current key
            new_key = base_key
        
        # Create musical parameters
        parameters = MusicalParameters(
            tempo=new_tempo,
            key=new_key,
            intensity=response["intensity"],
            instrumentation=response["instrumentation"],
            mood=response["mood"],
            transition_duration=response.get("duration", 10) / 2.0  # Half of total duration
        )
        
        logger.debug(f"Mapped event {event.type} to musical parameters: {parameters}")
        return parameters
    
    def _select_key(self, key_preference: str) -> str:
        """Select a musical key based on preference.
        
        Args:
            key_preference: Key preference (major, minor, diminished, neutral)
            
        Returns:
            Selected musical key
        """
        import random
        
        # Get list of keys for the preference
        keys = self.KEY_MAPPING.get(key_preference, self.KEY_MAPPING["neutral"])
        
        # Select a random key from the list
        selected_key = random.choice(keys)
        
        return selected_key
    
    @with_error_handling
    def get_immediate_trigger(self, event: MatchEvent) -> Dict[str, Any]:
        """Get immediate trigger information for a match event.
        
        This method provides information for immediate musical accents or triggers
        that should happen as soon as the event occurs.
        
        Args:
            event: Match event
            
        Returns:
            Dictionary with immediate trigger information
            
        Raises:
            MusicGenerationError: If trigger retrieval fails
        """
        # Get response configuration for the event type
        response = self.get_event_response(event.type)
        
        # Create trigger information
        trigger = {
            "type": response["immediate"],
            "intensity": response["intensity"],
            "duration": min(3.0, response["duration"] / 10.0)  # Short duration for immediate trigger
        }
        
        logger.debug(f"Generated immediate trigger for event {event.type}: {trigger}")
        return trigger
    
    @with_error_handling
    def enrich_parameters_with_context(
        self, 
        parameters: MusicalParameters, 
        match_context: Dict[str, Any]
    ) -> MusicalParameters:
        """Enrich musical parameters with match context information.
        
        This method adjusts musical parameters based on additional match context,
        such as score, time remaining, and match importance.
        
        Args:
            parameters: Base musical parameters
            match_context: Match context information
            
        Returns:
            Enriched musical parameters
            
        Raises:
            MusicGenerationError: If parameter enrichment fails
        """
        # Create a copy of parameters to modify
        from copy import deepcopy
        enriched = deepcopy(parameters)
        
        # Extract relevant context information
        score_diff = match_context.get("score_difference", 0)
        time_remaining = match_context.get("time_remaining", 45)
        match_importance = match_context.get("match_importance", 0.5)  # 0.0-1.0 scale
        
        # Adjust intensity based on match importance and time remaining
        if match_importance > 0.7:
            # Important match, increase intensity
            enriched.intensity = min(1.0, enriched.intensity * 1.2)
        
        if time_remaining < 10:
            # Late game, increase intensity
            enriched.intensity = min(1.0, enriched.intensity * 1.15)
        
        # Adjust tempo based on score difference
        if abs(score_diff) >= 3:
            # Blowout game, reduce tempo changes
            enriched.tempo = (enriched.tempo + parameters.tempo) / 2.0
        elif abs(score_diff) <= 1 and time_remaining < 15:
            # Close game in late stages, increase tempo
            enriched.tempo = min(180.0, enriched.tempo * 1.05)
        
        # Adjust transition duration based on match phase
        if time_remaining < 5:
            # Very late game, faster transitions
            enriched.transition_duration = max(1.0, enriched.transition_duration * 0.7)
        
        logger.debug(f"Enriched musical parameters with match context: {enriched}")
        return enriched