"""Musical accent triggering module.

This module provides functionality for triggering immediate musical accents
in response to significant events.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from note_seq.protobuf import music_pb2

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class AccentTrigger:
    """Manages musical accent triggering for significant events.
    
    This class provides methods for creating and triggering immediate musical
    accents in response to significant events like goals, cards, etc.
    """
    
    # Accent definitions for different event types
    ACCENT_DEFINITIONS = {
        "goal": {
            "intensity": 1.0,
            "duration": 2.0,
            "instruments": ["brass", "percussion"],
            "pattern": "fanfare",
            "priority": 10
        },
        "card": {
            "intensity": 0.7,
            "duration": 1.0,
            "instruments": ["strings", "percussion"],
            "pattern": "staccato",
            "priority": 7
        },
        "near_miss": {
            "intensity": 0.6,
            "duration": 1.5,
            "instruments": ["strings", "woodwinds"],
            "pattern": "rising",
            "priority": 5
        },
        "penalty": {
            "intensity": 0.8,
            "duration": 1.5,
            "instruments": ["percussion", "brass"],
            "pattern": "tension",
            "priority": 8
        },
        "corner": {
            "intensity": 0.4,
            "duration": 0.8,
            "instruments": ["strings"],
            "pattern": "short_rise",
            "priority": 3
        },
        "foul": {
            "intensity": 0.5,
            "duration": 0.7,
            "instruments": ["percussion"],
            "pattern": "hit",
            "priority": 4
        },
        "substitution": {
            "intensity": 0.3,
            "duration": 1.0,
            "instruments": ["woodwinds"],
            "pattern": "transition",
            "priority": 2
        }
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the accent trigger.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.sample_rate = 44100  # Default sample rate for audio output
        logger.info("Initialized AccentTrigger")
    
    @with_error_handling
    def create_accent_sequence(self, accent_type: str) -> music_pb2.NoteSequence:
        """Create a musical accent sequence for the specified event type.
        
        Args:
            accent_type: Type of accent to create (e.g., "goal", "card", "near_miss")
            
        Returns:
            NoteSequence with the accent
            
        Raises:
            MusicGenerationError: If accent creation fails
        """
        try:
            # Get accent definition
            definition = self.ACCENT_DEFINITIONS.get(accent_type)
            
            if not definition:
                # Use default accent if type not found
                logger.warning(f"Unknown accent type: {accent_type}, using default")
                return self._create_default_accent()
            
            # Create sequence based on pattern
            pattern = definition["pattern"]
            
            if pattern == "fanfare":
                return self._create_fanfare_accent(definition)
            elif pattern == "staccato":
                return self._create_staccato_accent(definition)
            elif pattern == "rising":
                return self._create_rising_accent(definition)
            elif pattern == "tension":
                return self._create_tension_accent(definition)
            elif pattern == "short_rise":
                return self._create_short_rise_accent(definition)
            elif pattern == "hit":
                return self._create_hit_accent(definition)
            elif pattern == "transition":
                return self._create_transition_accent(definition)
            else:
                # Use default accent if pattern not found
                logger.warning(f"Unknown accent pattern: {pattern}, using default")
                return self._create_default_accent()
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to create accent sequence: {str(e)}")
    
    @with_error_handling
    def create_accent_audio(self, accent_type: str) -> Tuple[np.ndarray, int]:
        """Create audio for the specified accent type.
        
        Args:
            accent_type: Type of accent to create (e.g., "goal", "card", "near_miss")
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
            
        Raises:
            MusicGenerationError: If audio creation fails
        """
        try:
            # Get accent definition
            definition = self.ACCENT_DEFINITIONS.get(accent_type)
            
            if not definition:
                # Use default accent if type not found
                logger.warning(f"Unknown accent type: {accent_type}, using default")
                return self._create_default_audio()
            
            # Create audio based on pattern
            pattern = definition["pattern"]
            intensity = definition["intensity"]
            duration = definition["duration"]
            
            if pattern == "fanfare":
                return self._create_fanfare_audio(intensity, duration)
            elif pattern == "staccato":
                return self._create_staccato_audio(intensity, duration)
            elif pattern == "rising":
                return self._create_rising_audio(intensity, duration)
            elif pattern == "tension":
                return self._create_tension_audio(intensity, duration)
            elif pattern == "short_rise":
                return self._create_short_rise_audio(intensity, duration)
            elif pattern == "hit":
                return self._create_hit_audio(intensity, duration)
            elif pattern == "transition":
                return self._create_transition_audio(intensity, duration)
            else:
                # Use default audio if pattern not found
                logger.warning(f"Unknown accent pattern: {pattern}, using default")
                return self._create_default_audio()
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to create accent audio: {str(e)}")
    
    @with_error_handling
    def get_accent_priority(self, accent_type: str) -> int:
        """Get the priority level of the specified accent type.
        
        Args:
            accent_type: Type of accent
            
        Returns:
            Priority level (higher = more important)
            
        Raises:
            MusicGenerationError: If priority retrieval fails
        """
        try:
            # Get accent definition
            definition = self.ACCENT_DEFINITIONS.get(accent_type)
            
            if not definition:
                # Use default priority if type not found
                return 1
            
            return definition["priority"]
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to get accent priority: {str(e)}")
    
    def _create_fanfare_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a fanfare accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with fanfare accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a triumphant brass fanfare
        # C major chord with ascending pattern
        notes = [60, 64, 67, 72, 76, 79, 84]
        start_time = 0.0
        
        for note_pitch in notes:
            note = sequence.notes.add()
            note.start_time = start_time
            note.end_time = start_time + 0.25
            note.pitch = note_pitch
            note.velocity = 100
            note.instrument = 0
            note.program = 61  # Brass
            start_time += 0.125
        
        # Add cymbal crash
        cymbal = sequence.notes.add()
        cymbal.start_time = 0.0
        cymbal.end_time = 1.0
        cymbal.pitch = 57  # Cymbal
        cymbal.velocity = 120
        cymbal.instrument = 1
        cymbal.program = 128  # Percussion
        
        sequence.total_time = max(start_time, 1.0)
        return sequence
    
    def _create_staccato_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a staccato accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with staccato accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a tense, dissonant chord
        # Diminished chord
        notes = [60, 63, 66, 69]
        
        for note_pitch in notes:
            note = sequence.notes.add()
            note.start_time = 0.0
            note.end_time = 0.5
            note.pitch = note_pitch
            note.velocity = 90
            note.instrument = 0
            note.program = 48  # Strings
        
        sequence.total_time = 0.5
        return sequence
    
    def _create_rising_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a rising accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with rising accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a rising string figure that cuts off
        notes = [60, 62, 64, 65, 67, 69, 71]
        start_time = 0.0
        
        for i, note_pitch in enumerate(notes):
            note = sequence.notes.add()
            note.start_time = start_time
            # Last note cuts off
            note.end_time = start_time + (0.2 if i < len(notes) - 1 else 0.1)
            note.pitch = note_pitch
            note.velocity = 80 + i * 5  # Crescendo
            note.instrument = 0
            note.program = 48  # Strings
            start_time += 0.1
        
        sequence.total_time = start_time
        return sequence
    
    def _create_tension_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a tension accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with tension accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a tense percussion roll
        for i in range(8):
            note = sequence.notes.add()
            note.start_time = i * 0.125
            note.end_time = i * 0.125 + 0.1
            note.pitch = 60 - (i % 3) * 2  # Alternating pitches
            note.velocity = 70 + i * 5  # Crescendo
            note.instrument = 1
            note.program = 128  # Percussion
        
        sequence.total_time = 1.0
        return sequence
    
    def _create_short_rise_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a short rise accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with short rise accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a short rising figure
        notes = [60, 64, 67, 72]
        start_time = 0.0
        
        for note_pitch in notes:
            note = sequence.notes.add()
            note.start_time = start_time
            note.end_time = start_time + 0.15
            note.pitch = note_pitch
            note.velocity = 80
            note.instrument = 0
            note.program = 48  # Strings
            start_time += 0.1
        
        sequence.total_time = start_time
        return sequence
    
    def _create_hit_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a hit accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with hit accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a percussion hit
        note = sequence.notes.add()
        note.start_time = 0.0
        note.end_time = 0.3
        note.pitch = 60
        note.velocity = 100
        note.instrument = 1
        note.program = 128  # Percussion
        
        sequence.total_time = 0.3
        return sequence
    
    def _create_transition_accent(self, definition: Dict[str, Any]) -> music_pb2.NoteSequence:
        """Create a transition accent sequence.
        
        Args:
            definition: Accent definition
            
        Returns:
            NoteSequence with transition accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a woodwind transition figure
        notes = [72, 71, 69, 67, 65, 64, 62, 60]
        start_time = 0.0
        
        for note_pitch in notes:
            note = sequence.notes.add()
            note.start_time = start_time
            note.end_time = start_time + 0.2
            note.pitch = note_pitch
            note.velocity = 70
            note.instrument = 0
            note.program = 74  # Flute
            start_time += 0.125
        
        sequence.total_time = start_time
        return sequence
    
    def _create_default_accent(self) -> music_pb2.NoteSequence:
        """Create a default accent sequence.
        
        Returns:
            NoteSequence with default accent
        """
        sequence = music_pb2.NoteSequence()
        
        # Add a simple chord
        notes = [60, 64, 67]
        
        for note_pitch in notes:
            note = sequence.notes.add()
            note.start_time = 0.0
            note.end_time = 0.5
            note.pitch = note_pitch
            note.velocity = 80
            note.instrument = 0
            note.program = 0  # Piano
        
        sequence.total_time = 0.5
        return sequence
    
    def _create_fanfare_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a fanfare accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a fanfare-like sound with multiple frequencies
        audio = np.sin(2 * np.pi * 440 * t)  # Base note
        audio += 0.7 * np.sin(2 * np.pi * 554 * t)  # Major third
        audio += 0.5 * np.sin(2 * np.pi * 659 * t)  # Fifth
        audio += 0.3 * np.sin(2 * np.pi * 880 * t)  # Octave
        
        # Add some noise for cymbal crash
        noise = np.random.normal(0, 0.2, len(t))
        noise *= np.exp(-5 * t)  # Decay
        
        audio += noise
        
        # Apply envelope
        envelope = np.ones_like(t)
        attack = int(0.05 * self.sample_rate)
        decay = int(0.9 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[decay:] = np.linspace(1, 0, len(envelope) - decay)
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_staccato_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a staccato accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a staccato-like sound with dissonant frequencies
        audio = np.sin(2 * np.pi * 440 * t)  # Base note
        audio += 0.7 * np.sin(2 * np.pi * 466 * t)  # Minor second (dissonant)
        audio += 0.5 * np.sin(2 * np.pi * 622 * t)  # Diminished fifth (dissonant)
        
        # Apply staccato envelope
        envelope = np.zeros_like(t)
        attack = int(0.01 * self.sample_rate)
        decay = int(0.1 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0, decay)
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_rising_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a rising accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a rising frequency sweep
        freq = 440 + 440 * t / duration  # 440 Hz to 880 Hz
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        audio = np.sin(phase)
        
        # Apply envelope
        envelope = np.ones_like(t)
        attack = int(0.05 * self.sample_rate)
        decay = int(0.9 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[decay:] = np.linspace(1, 0, len(envelope) - decay)
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_tension_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a tension accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a tension-like sound with tremolo effect
        freq = 220
        tremolo_rate = 8  # Hz
        tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * tremolo_rate * t)
        audio = tremolo * np.sin(2 * np.pi * freq * t)
        
        # Add some noise for percussion
        noise = np.random.normal(0, 0.2, len(t))
        noise *= np.sin(2 * np.pi * 8 * t)**2  # Pulsing noise
        
        audio += noise
        
        # Apply crescendo envelope
        envelope = np.linspace(0.3, 1, len(t))
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_short_rise_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a short rise accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a short rising frequency sweep
        freq = 440 + 220 * t / duration  # 440 Hz to 660 Hz
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        audio = np.sin(phase)
        
        # Apply envelope
        envelope = np.zeros_like(t)
        attack = int(0.05 * self.sample_rate)
        decay = int(0.3 * self.sample_rate)
        if attack + decay > len(envelope):
            decay = len(envelope) - attack
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0, decay)
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_hit_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a hit accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a percussion hit with noise
        noise = np.random.normal(0, 1, len(t))
        
        # Apply sharp decay envelope
        envelope = np.exp(-10 * t)
        
        audio = noise * envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_transition_audio(self, intensity: float, duration: float) -> Tuple[np.ndarray, int]:
        """Create audio for a transition accent.
        
        Args:
            intensity: Accent intensity (0-1)
            duration: Accent duration in seconds
            
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a descending frequency sweep
        freq = 880 - 440 * t / duration  # 880 Hz to 440 Hz
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        audio = np.sin(phase)
        
        # Apply envelope
        envelope = np.ones_like(t)
        attack = int(0.05 * self.sample_rate)
        decay = int(0.8 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[decay:] = np.linspace(1, 0, len(envelope) - decay)
        
        audio *= envelope
        
        # Normalize and scale by intensity
        audio = audio / np.max(np.abs(audio)) * intensity
        
        return audio, self.sample_rate
    
    def _create_default_audio(self) -> Tuple[np.ndarray, int]:
        """Create audio for a default accent.
        
        Returns:
            Tuple of (audio_buffer, sample_rate)
        """
        # Generate simple audio for testing
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create a simple chord
        audio = np.sin(2 * np.pi * 440 * t)  # C
        audio += np.sin(2 * np.pi * 554 * t)  # E
        audio += np.sin(2 * np.pi * 659 * t)  # G
        
        # Apply envelope
        envelope = np.exp(-5 * t)
        
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        return audio, self.sample_rate