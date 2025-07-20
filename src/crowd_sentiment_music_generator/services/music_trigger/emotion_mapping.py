"""Emotion-to-music mapping module.

This module provides functionality for mapping crowd emotions to musical parameters,
including intensity scaling and cultural adaptation.
"""

import logging
from typing import Dict, Any, Optional, List

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.data.user_preferences import UserPreferences
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class EmotionMusicMapper:
    """Maps crowd emotions to musical parameters.
    
    This class provides methods for mapping different emotions to appropriate
    musical parameters, with intensity scaling and cultural adaptation.
    """
    
    # Emotion musical mapping configuration
    EMOTION_MUSICAL_MAPPING = {
        "excitement": {
            "tempo_factor": 1.2,  # Increase tempo by 20%
            "intensity_factor": 1.0,  # Full intensity
            "key_preference": "major",
            "instrumentation": ["brass", "percussion", "strings"],
            "mood": "energetic",
            "transition_speed": "fast"
        },
        "joy": {
            "tempo_factor": 1.1,  # Increase tempo by 10%
            "intensity_factor": 0.8,  # 80% intensity
            "key_preference": "major",
            "instrumentation": ["strings", "woodwinds", "piano"],
            "mood": "uplifting",
            "transition_speed": "moderate"
        },
        "tension": {
            "tempo_factor": 0.9,  # Decrease tempo by 10%
            "intensity_factor": 0.7,  # 70% intensity
            "key_preference": "minor",
            "instrumentation": ["strings", "percussion", "brass"],
            "mood": "suspenseful",
            "transition_speed": "moderate"
        },
        "disappointment": {
            "tempo_factor": 0.8,  # Decrease tempo by 20%
            "intensity_factor": 0.5,  # 50% intensity
            "key_preference": "minor",
            "instrumentation": ["strings", "piano"],
            "mood": "somber",
            "transition_speed": "slow"
        },
        "anger": {
            "tempo_factor": 1.1,  # Increase tempo by 10%
            "intensity_factor": 0.9,  # 90% intensity
            "key_preference": "diminished",
            "instrumentation": ["percussion", "brass", "electric_guitar"],
            "mood": "intense",
            "transition_speed": "fast"
        },
        "anticipation": {
            "tempo_factor": 1.05,  # Increase tempo by 5%
            "intensity_factor": 0.6,  # 60% intensity
            "key_preference": "neutral",
            "instrumentation": ["strings", "woodwinds", "percussion"],
            "mood": "building",
            "transition_speed": "moderate"
        },
        "neutral": {
            "tempo_factor": 1.0,  # No change
            "intensity_factor": 0.4,  # 40% intensity
            "key_preference": "neutral",
            "instrumentation": ["strings", "piano"],
            "mood": "ambient",
            "transition_speed": "slow"
        }
    }
    
    # Cultural adaptation configurations
    CULTURAL_ADAPTATIONS = {
        "global": {
            # Default global settings (no specific adaptation)
            "instrumentation_bias": {},
            "tempo_bias": 1.0,
            "key_bias": {}
        },
        "european": {
            "instrumentation_bias": {
                "strings": 1.2,
                "brass": 1.1,
                "woodwinds": 1.2,
                "piano": 1.1
            },
            "tempo_bias": 0.95,  # Slightly slower
            "key_bias": {
                "major": 1.1,
                "minor": 1.1
            }
        },
        "latin": {
            "instrumentation_bias": {
                "percussion": 1.3,
                "acoustic_guitar": 1.4,
                "brass": 1.2
            },
            "tempo_bias": 1.1,  # Slightly faster
            "key_bias": {
                "major": 1.2,
                "minor": 0.9
            }
        },
        "asian": {
            "instrumentation_bias": {
                "strings": 1.1,
                "woodwinds": 1.3,
                "percussion": 1.2
            },
            "tempo_bias": 1.0,  # No change
            "key_bias": {
                "pentatonic": 1.5
            }
        },
        "african": {
            "instrumentation_bias": {
                "percussion": 1.5,
                "strings": 0.9,
                "woodwinds": 1.2
            },
            "tempo_bias": 1.05,  # Slightly faster
            "key_bias": {
                "major": 1.1,
                "minor": 0.9
            }
        },
        "middle_eastern": {
            "instrumentation_bias": {
                "strings": 1.2,
                "woodwinds": 1.3,
                "percussion": 1.2
            },
            "tempo_bias": 0.95,  # Slightly slower
            "key_bias": {
                "minor": 1.3,
                "diminished": 1.2
            }
        }
    }
    
    # Transition speed mappings (in seconds)
    TRANSITION_SPEEDS = {
        "fast": 2.0,
        "moderate": 4.0,
        "slow": 6.0
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the emotion music mapper.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.base_tempo = 100.0  # Default base tempo in BPM
        self.current_key = "C Major"  # Default key
        self.cultural_style = "global"  # Default cultural style
        logger.info("Initialized EmotionMusicMapper")
    
    def set_base_musical_state(self, tempo: float, key: str) -> None:
        """Set the base musical state for emotion mapping.
        
        Args:
            tempo: Base tempo in BPM
            key: Base musical key
        """
        self.base_tempo = tempo
        self.current_key = key
        logger.debug(f"Set base musical state: tempo={tempo} BPM, key={key}")
    
    def set_cultural_style(self, style: str) -> None:
        """Set the cultural adaptation style.
        
        Args:
            style: Cultural style (european, latin, asian, african, middle_eastern, global)
        """
        if style in self.CULTURAL_ADAPTATIONS:
            self.cultural_style = style
        else:
            self.cultural_style = "global"
        logger.debug(f"Set cultural style: {self.cultural_style}")
    
    @with_error_handling
    def map_emotion_to_parameters(
        self, 
        emotion: CrowdEmotion, 
        current_parameters: Optional[MusicalParameters] = None
    ) -> MusicalParameters:
        """Map a crowd emotion to musical parameters.
        
        This method generates appropriate musical parameters based on the emotion type,
        intensity, and current musical state.
        
        Args:
            emotion: Crowd emotion to map
            current_parameters: Current musical parameters (optional)
            
        Returns:
            MusicalParameters object with parameters for the emotion
            
        Raises:
            MusicGenerationError: If parameter generation fails
        """
        # Get mapping configuration for the emotion type
        emotion_type = emotion.emotion
        mapping = self.EMOTION_MUSICAL_MAPPING.get(emotion_type, self.EMOTION_MUSICAL_MAPPING["neutral"])
        
        # Use current parameters as base if provided, otherwise use defaults
        base_tempo = current_parameters.tempo if current_parameters else self.base_tempo
        base_key = current_parameters.key if current_parameters else self.current_key
        
        # Calculate new tempo based on emotion mapping and intensity
        tempo_factor = mapping["tempo_factor"]
        intensity_normalized = emotion.intensity / 100.0  # Convert to 0-1 scale
        
        # Scale tempo factor based on intensity
        # At full intensity, use the full factor; at zero intensity, use no change
        scaled_tempo_factor = 1.0 + ((tempo_factor - 1.0) * intensity_normalized)
        
        # Apply cultural adaptation to tempo
        cultural_tempo_bias = self.CULTURAL_ADAPTATIONS[self.cultural_style]["tempo_bias"]
        new_tempo = base_tempo * scaled_tempo_factor * cultural_tempo_bias
        
        # Ensure tempo stays within reasonable bounds
        new_tempo = max(60.0, min(180.0, new_tempo))
        
        # Select key based on emotion preference and cultural adaptation
        key_preference = mapping["key_preference"]
        new_key = self._select_key_for_emotion(key_preference, base_key)
        
        # Apply cultural adaptation to instrumentation
        instrumentation = self._adapt_instrumentation_to_culture(mapping["instrumentation"])
        
        # Calculate transition duration based on emotion
        transition_speed = mapping["transition_speed"]
        transition_duration = self.TRANSITION_SPEEDS[transition_speed]
        
        # Scale intensity based on emotion mapping and confidence
        intensity = intensity_normalized * mapping["intensity_factor"]
        
        # Adjust intensity based on confidence
        if emotion.confidence < 0.7:
            # Lower confidence means more conservative intensity changes
            intensity = (intensity + (current_parameters.intensity if current_parameters else 0.5)) / 2.0
        
        # Create musical parameters
        parameters = MusicalParameters(
            tempo=new_tempo,
            key=new_key,
            intensity=intensity,
            instrumentation=instrumentation,
            mood=mapping["mood"],
            transition_duration=transition_duration
        )
        
        logger.debug(f"Mapped emotion {emotion_type} to musical parameters: {parameters}")
        return parameters
    
    def _select_key_for_emotion(self, key_preference: str, current_key: str) -> str:
        """Select a musical key based on emotion preference and current key.
        
        Args:
            key_preference: Key preference (major, minor, diminished, neutral)
            current_key: Current musical key
            
        Returns:
            Selected musical key
        """
        import random
        
        # If neutral preference, keep current key
        if key_preference == "neutral":
            return current_key
        
        # Get cultural key bias
        cultural_key_bias = self.CULTURAL_ADAPTATIONS[self.cultural_style].get("key_bias", {})
        
        # Check if there's a cultural bias for this key preference
        bias = cultural_key_bias.get(key_preference, 1.0)
        
        # If bias is strong enough, apply the preference
        if bias > 1.0 or random.random() < bias:
            # Get list of keys for the preference
            from crowd_sentiment_music_generator.services.music_trigger.event_mapping import EventMusicMapper
            keys = EventMusicMapper.KEY_MAPPING.get(key_preference, EventMusicMapper.KEY_MAPPING["neutral"])
            
            # Select a random key from the list
            selected_key = random.choice(keys)
            return selected_key
        else:
            # Keep current key
            return current_key
    
    def _adapt_instrumentation_to_culture(self, base_instrumentation: List[str]) -> List[str]:
        """Adapt instrumentation based on cultural style.
        
        Args:
            base_instrumentation: Base instrumentation list
            
        Returns:
            Culturally adapted instrumentation list
        """
        # Get cultural instrumentation bias
        cultural_instr_bias = self.CULTURAL_ADAPTATIONS[self.cultural_style].get("instrumentation_bias", {})
        
        if not cultural_instr_bias:
            return base_instrumentation
        
        # Start with base instrumentation
        adapted_instrumentation = list(base_instrumentation)
        
        # Add culturally biased instruments
        for instrument, bias in cultural_instr_bias.items():
            if bias > 1.1 and instrument not in adapted_instrumentation:
                adapted_instrumentation.append(instrument)
        
        return adapted_instrumentation
    
    @with_error_handling
    def scale_parameters_with_intensity(
        self, 
        parameters: MusicalParameters, 
        intensity: float
    ) -> MusicalParameters:
        """Scale musical parameters based on intensity level.
        
        This method adjusts various musical parameters proportionally to the
        provided intensity level.
        
        Args:
            parameters: Base musical parameters
            intensity: Intensity level (0-100)
            
        Returns:
            Scaled musical parameters
            
        Raises:
            MusicGenerationError: If parameter scaling fails
        """
        # Create a copy of parameters to modify
        from copy import deepcopy
        scaled = deepcopy(parameters)
        
        # Normalize intensity to 0-1 scale
        intensity_normalized = intensity / 100.0
        
        # Scale tempo: higher intensity = faster tempo (up to 20% increase)
        tempo_factor = 1.0 + (0.2 * intensity_normalized)
        scaled.tempo = min(180.0, parameters.tempo * tempo_factor)
        
        # Scale intensity directly
        scaled.intensity = intensity_normalized
        
        # Adjust instrumentation based on intensity
        if intensity_normalized > 0.7:
            # Add percussion for high intensity if not already present
            if "percussion" not in scaled.instrumentation:
                scaled.instrumentation.append("percussion")
            
            # Add brass for very high intensity if not already present
            if intensity_normalized > 0.9 and "brass" not in scaled.instrumentation:
                scaled.instrumentation.append("brass")
        elif intensity_normalized < 0.3:
            # Remove percussion and brass for low intensity
            scaled.instrumentation = [i for i in scaled.instrumentation 
                                     if i not in ["percussion", "brass"]]
            
            # Ensure strings or piano for low intensity
            if "strings" not in scaled.instrumentation and "piano" not in scaled.instrumentation:
                scaled.instrumentation.append("strings")
        
        # Adjust transition duration: higher intensity = faster transitions
        if scaled.transition_duration is not None:
            transition_factor = 1.0 - (0.5 * intensity_normalized)  # Up to 50% faster
            scaled.transition_duration = max(1.0, scaled.transition_duration * transition_factor)
        
        logger.debug(f"Scaled parameters with intensity {intensity}: {scaled}")
        return scaled
    
    @with_error_handling
    def apply_user_preferences(
        self, 
        parameters: MusicalParameters, 
        preferences: UserPreferences
    ) -> MusicalParameters:
        """Apply user preferences to musical parameters.
        
        This method adjusts musical parameters based on user preferences,
        such as preferred genres, intensity levels, and cultural style.
        
        Args:
            parameters: Base musical parameters
            preferences: User preferences
            
        Returns:
            Adjusted musical parameters
            
        Raises:
            MusicGenerationError: If preference application fails
        """
        # Create a copy of parameters to modify
        from copy import deepcopy
        adjusted = deepcopy(parameters)
        
        # Apply intensity preference (1-5 scale)
        intensity_factor = preferences.music_intensity / 3.0  # Scale to around 1.0 at middle setting (3)
        adjusted.intensity = min(1.0, parameters.intensity * intensity_factor)
        
        # Apply genre preferences
        if preferences.preferred_genres:
            # Adjust instrumentation based on preferred genres
            if "orchestral" in preferences.preferred_genres:
                # Ensure full orchestral instrumentation
                for instrument in ["strings", "brass", "woodwinds", "percussion"]:
                    if instrument not in adjusted.instrumentation:
                        adjusted.instrumentation.append(instrument)
            
            elif "electronic" in preferences.preferred_genres:
                # Add electronic instruments
                electronic_instruments = ["synth", "electronic_percussion", "bass"]
                for instrument in electronic_instruments:
                    if instrument not in adjusted.instrumentation:
                        adjusted.instrumentation.append(instrument)
                
                # Remove some acoustic instruments
                adjusted.instrumentation = [i for i in adjusted.instrumentation 
                                          if i not in ["woodwinds"]]
            
            elif "acoustic" in preferences.preferred_genres:
                # Focus on acoustic instruments
                adjusted.instrumentation = [i for i in adjusted.instrumentation 
                                          if i not in ["synth", "electronic_percussion"]]
                
                # Add acoustic instruments
                acoustic_instruments = ["acoustic_guitar", "piano", "strings"]
                for instrument in acoustic_instruments:
                    if instrument not in adjusted.instrumentation:
                        adjusted.instrumentation.append(instrument)
        
        # Apply cultural style preference
        if preferences.cultural_style:
            # Set cultural style for future mappings
            self.set_cultural_style(preferences.cultural_style)
            
            # Apply cultural adaptation to current parameters
            cultural_instr_bias = self.CULTURAL_ADAPTATIONS[self.cultural_style].get("instrumentation_bias", {})
            
            # Add culturally preferred instruments
            for instrument, bias in cultural_instr_bias.items():
                if bias > 1.2 and instrument not in adjusted.instrumentation:
                    adjusted.instrumentation.append(instrument)
        
        logger.debug(f"Applied user preferences to parameters: {adjusted}")
        return adjusted