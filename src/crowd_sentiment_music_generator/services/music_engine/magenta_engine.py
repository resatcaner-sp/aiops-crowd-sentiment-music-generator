"""Magenta real-time music engine module.

This module provides the MagentaMusicEngine class that generates real-time music
based on emotional inputs using Google's Magenta models.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import note_seq

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class MagentaMusicEngine:
    """Generates real-time music based on emotional inputs using Magenta models.
    
    This class provides methods for initializing Magenta models, generating music,
    and evolving musical compositions based on emotional inputs. It uses the
    Performance RNN model from Google's Magenta project to generate MIDI-like
    musical sequences that can be rendered to audio in real-time.
    """
    
    # Model bundle paths
    MODEL_BUNDLES = {
        "performance_rnn": "performance_with_dynamics.mag",
        "performance_rnn_conditional": "performance_with_dynamics_and_modulations.mag",
        "melody_rnn": "attention_rnn.mag"
    }
    
    # Key mappings for different emotions
    KEY_MAPPINGS = {
        "major": ["C", "G", "D", "A", "E", "F"],
        "minor": ["Am", "Em", "Dm", "Gm", "Cm", "Fm"],
        "diminished": ["Bdim", "Fdim", "Cdim"],
        "neutral": ["C", "G", "Am", "Em"]
    }
    
    # Base melodies for different contexts
    BASE_MELODIES = {
        "neutral": [
            [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
            [69, 67, 65, 64, 62, 60, 59, 60]   # Descending melody
        ],
        "exciting": [
            [60, 64, 67, 72, 67, 64, 60],      # C major arpeggio
            [60, 62, 64, 67, 69, 71, 72]       # Ascending scale
        ],
        "tense": [
            [60, 63, 66, 69, 66, 63, 60],      # C minor arpeggio
            [60, 61, 63, 66, 68, 70, 72]       # Minor scale
        ],
        "sad": [
            [69, 67, 65, 64, 62, 60],          # Descending melody
            [60, 63, 65, 67, 63, 60]           # Minor pattern
        ]
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the Magenta music engine.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.models_path = self.config.models_path
        self.generator = None
        self.current_sequence = None
        self.current_parameters = None
        self.generation_start_time = 0
        self.is_initialized = False
        self.audio_buffer = np.zeros(0)
        self.sample_rate = 44100  # Default sample rate for audio output
        logger.info("Initialized MagentaMusicEngine")
    
    @with_error_handling
    def initialize_model(self, model_type: str = "performance_rnn") -> None:
        """Initialize the Magenta model.
        
        Args:
            model_type: Type of model to initialize (performance_rnn, performance_rnn_conditional, melody_rnn)
            
        Raises:
            MusicGenerationError: If model initialization fails
        """
        if model_type not in self.MODEL_BUNDLES:
            raise MusicGenerationError(f"Unknown model type: {model_type}")
        
        try:
            bundle_file = os.path.join(self.models_path, self.MODEL_BUNDLES[model_type])
            
            # Check if model file exists
            if not os.path.exists(bundle_file):
                # For testing/development, use a dummy generator
                logger.warning(f"Model file not found: {bundle_file}. Using dummy generator.")
                self.generator = DummySequenceGenerator()
                self.is_initialized = True
                return
            
            # Load the bundle
            bundle = sequence_generator_bundle.read_bundle_file(bundle_file)
            
            # Initialize the generator based on model type
            if model_type.startswith("performance_rnn"):
                self.generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
                    model=performance_sequence_generator.PerformanceRnnModel(bundle),
                    details=bundle.generator_details,
                    steps_per_quarter=4,
                    num_velocity_bins=32
                )
            else:
                # For other model types (to be implemented)
                raise MusicGenerationError(f"Model type not implemented: {model_type}")
            
            self.is_initialized = True
            logger.info(f"Initialized Magenta model: {model_type}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to initialize Magenta model: {str(e)}")
    
    @with_error_handling
    def initialize_base_melody(self, match_context: Optional[Dict[str, Any]] = None) -> None:
        """Start with appropriate base theme for the match.
        
        Args:
            match_context: Optional match context information
            
        Raises:
            MusicGenerationError: If base melody initialization fails
        """
        if not self.is_initialized:
            self.initialize_model()
        
        try:
            # Determine appropriate base melody based on match context
            melody_type = "neutral"
            
            if match_context:
                # Select melody based on match context
                if match_context.get("match_importance", 0) > 0.7:
                    melody_type = "exciting"
                elif match_context.get("score_difference", 0) > 2:
                    melody_type = "exciting" if match_context.get("winning_team") == "home" else "tense"
                elif match_context.get("time_remaining", 90) < 10:
                    melody_type = "tense"
            
            # Select a random base melody from the appropriate category
            import random
            base_notes = random.choice(self.BASE_MELODIES[melody_type])
            
            # Create a sequence with the base melody
            sequence = music_pb2.NoteSequence()
            
            # Add notes to the sequence
            start_time = 0.0
            for note_pitch in base_notes:
                note = sequence.notes.add()
                note.start_time = start_time
                note.end_time = start_time + 0.5
                note.pitch = note_pitch
                note.velocity = 80
                note.instrument = 0
                note.program = 0
                start_time += 0.5
            
            sequence.total_time = start_time
            
            # Set as current sequence
            self.current_sequence = sequence
            self.generation_start_time = time.time()
            
            logger.info(f"Initialized base melody of type: {melody_type}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to initialize base melody: {str(e)}")
    
    @with_error_handling
    def evolve_music(self, params: MusicalParameters) -> None:
        """Evolve the music based on new parameters.
        
        Args:
            params: Musical parameters to apply
            
        Raises:
            MusicGenerationError: If music evolution fails
        """
        if not self.is_initialized:
            self.initialize_model()
        
        if not self.current_sequence:
            self.initialize_base_melody()
        
        try:
            # Store current parameters
            self.current_parameters = params
            
            # Create generation options based on parameters
            temperature = self._map_intensity_to_temperature(params.intensity)
            
            # Create generator options
            generator_options = generator_pb2.GeneratorOptions()
            generator_options.args['temperature'].float_value = temperature
            
            # Set generation length based on tempo
            # Faster tempo = shorter generation to allow more frequent updates
            seconds_per_step = 60.0 / params.tempo / 4  # 4 steps per beat
            num_steps = int(4 * 4 * 4)  # 4 bars of 4 beats at 4 steps per beat
            
            # Generate continuation
            if isinstance(self.generator, DummySequenceGenerator):
                # Use dummy generator for testing
                continuation = self.generator.generate(
                    self.current_sequence, generator_options, num_steps=num_steps
                )
            else:
                # Use actual Magenta generator
                continuation = self.generator.generate(
                    self.current_sequence, generator_options
                )
            
            # Update current sequence
            self.current_sequence = continuation
            self.generation_start_time = time.time()
            
            logger.info(f"Evolved music with parameters: tempo={params.tempo}, key={params.key}, "
                       f"intensity={params.intensity}, mood={params.mood}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to evolve music: {str(e)}")
    
    @with_error_handling
    def trigger_accent(self, accent_type: str) -> None:
        """Trigger an immediate musical accent.
        
        Args:
            accent_type: Type of accent to trigger (e.g., "goal", "card", "near_miss")
            
        Raises:
            MusicGenerationError: If accent triggering fails
        """
        if not self.is_initialized:
            self.initialize_model()
        
        try:
            # Define accent patterns for different types
            accent_patterns = {
                "goal": self._create_goal_accent(),
                "card": self._create_card_accent(),
                "near_miss": self._create_near_miss_accent(),
                "penalty": self._create_penalty_accent(),
                "default": self._create_default_accent()
            }
            
            # Get the appropriate accent pattern
            accent_sequence = accent_patterns.get(accent_type, accent_patterns["default"])
            
            # Merge accent with current sequence
            if self.current_sequence:
                # Find the current end time
                end_time = self.current_sequence.total_time
                
                # Shift accent to start at current end time
                for note in accent_sequence.notes:
                    note.start_time += end_time
                    note.end_time += end_time
                
                # Merge sequences
                merged_sequence = music_pb2.NoteSequence()
                merged_sequence.CopyFrom(self.current_sequence)
                
                for note in accent_sequence.notes:
                    new_note = merged_sequence.notes.add()
                    new_note.CopyFrom(note)
                
                merged_sequence.total_time = accent_sequence.total_time + end_time
                
                # Update current sequence
                self.current_sequence = merged_sequence
            else:
                # If no current sequence, just use the accent
                self.current_sequence = accent_sequence
            
            logger.info(f"Triggered accent: {accent_type}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to trigger accent: {str(e)}")
    
    @with_error_handling
    def transition_to(self, target_params: MusicalParameters, duration: float) -> None:
        """Smoothly transition to new musical parameters.
        
        Args:
            target_params: Target musical parameters
            duration: Transition duration in seconds
            
        Raises:
            MusicGenerationError: If transition fails
        """
        if not self.is_initialized:
            self.initialize_model()
        
        if not self.current_parameters:
            # If no current parameters, just set directly
            self.evolve_music(target_params)
            return
        
        try:
            # Calculate number of steps for smooth transition
            steps = max(2, int(duration / 0.5))  # At least 2 steps, update every 0.5 seconds
            
            # Create intermediate parameter steps
            for i in range(steps):
                # Calculate interpolation factor (0 to 1)
                t = (i + 1) / steps
                
                # Create interpolated parameters
                interpolated = self._interpolate_parameters(self.current_parameters, target_params, t)
                
                # Evolve music with interpolated parameters
                self.evolve_music(interpolated)
                
                # In a real-time system, we would wait here
                # For testing, we'll just continue
            
            logger.info(f"Completed transition to new parameters over {duration} seconds")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to transition: {str(e)}")
    
    @with_error_handling
    def get_audio_output(self) -> Tuple[np.ndarray, int]:
        """Get the current audio output buffer.
        
        Returns:
            Tuple of (audio_buffer, sample_rate)
            
        Raises:
            MusicGenerationError: If audio generation fails
        """
        if not self.is_initialized or not self.current_sequence:
            # Return empty buffer if not initialized
            return np.zeros(0), self.sample_rate
        
        try:
            # In a real implementation, this would synthesize audio from the sequence
            # For now, we'll generate a simple sine wave based on the parameters
            
            if not self.current_parameters:
                return np.zeros(0), self.sample_rate
            
            # Generate 1 second of audio
            duration = 1.0
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            
            # Base frequency based on key (C4 = 261.63 Hz)
            base_freq = 261.63
            
            # Simple sine wave with harmonics
            audio = np.sin(2 * np.pi * base_freq * t)
            
            # Add harmonics based on intensity
            if self.current_parameters.intensity > 0.3:
                audio += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
            
            if self.current_parameters.intensity > 0.6:
                audio += 0.3 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Scale by intensity
            audio = audio * self.current_parameters.intensity
            
            # Update buffer
            self.audio_buffer = audio
            
            return audio, self.sample_rate
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to generate audio: {str(e)}")
    
    def _map_intensity_to_temperature(self, intensity: float) -> float:
        """Map intensity value to temperature parameter for Magenta.
        
        Args:
            intensity: Intensity value (0-1)
            
        Returns:
            Temperature value for Magenta generator
        """
        # Higher intensity = lower temperature (more predictable, energetic)
        # Lower intensity = higher temperature (more random, ambient)
        min_temp = 0.8
        max_temp = 1.5
        
        # Inverse mapping
        temperature = max_temp - (intensity * (max_temp - min_temp))
        
        return temperature
    
    def _interpolate_parameters(
        self, 
        start_params: MusicalParameters, 
        end_params: MusicalParameters, 
        t: float
    ) -> MusicalParameters:
        """Interpolate between two sets of parameters.
        
        Args:
            start_params: Starting parameters
            end_params: Ending parameters
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated parameters
        """
        # Create a new parameter object
        from copy import deepcopy
        params = deepcopy(start_params)
        
        # Interpolate numeric parameters
        params.tempo = start_params.tempo + t * (end_params.tempo - start_params.tempo)
        params.intensity = start_params.intensity + t * (end_params.intensity - start_params.intensity)
        
        # For key, use end key if t > 0.5, otherwise start key
        params.key = end_params.key if t > 0.5 else start_params.key
        
        # For mood, use end mood if t > 0.5, otherwise start mood
        params.mood = end_params.mood if t > 0.5 else start_params.mood
        
        # For instrumentation, gradually add end instruments and remove start instruments
        if t < 0.33:
            # Keep start instrumentation
            params.instrumentation = start_params.instrumentation
        elif t < 0.67:
            # Mix instrumentation
            params.instrumentation = list(set(start_params.instrumentation + end_params.instrumentation))
        else:
            # Use end instrumentation
            params.instrumentation = end_params.instrumentation
        
        # For transition duration, use weighted average
        if start_params.transition_duration is not None and end_params.transition_duration is not None:
            params.transition_duration = (
                start_params.transition_duration + 
                t * (end_params.transition_duration - start_params.transition_duration)
            )
        
        return params
    
    def _create_goal_accent(self) -> music_pb2.NoteSequence:
        """Create a musical accent for a goal.
        
        Returns:
            NoteSequence with goal accent
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
    
    def _create_card_accent(self) -> music_pb2.NoteSequence:
        """Create a musical accent for a card.
        
        Returns:
            NoteSequence with card accent
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
    
    def _create_near_miss_accent(self) -> music_pb2.NoteSequence:
        """Create a musical accent for a near miss.
        
        Returns:
            NoteSequence with near miss accent
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
    
    def _create_penalty_accent(self) -> music_pb2.NoteSequence:
        """Create a musical accent for a penalty.
        
        Returns:
            NoteSequence with penalty accent
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
    
    def _create_default_accent(self) -> music_pb2.NoteSequence:
        """Create a default musical accent.
        
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


class DummySequenceGenerator:
    """Dummy sequence generator for testing without Magenta models."""
    
    def generate(
        self, 
        input_sequence: music_pb2.NoteSequence, 
        generator_options: Any, 
        num_steps: int = 128
    ) -> music_pb2.NoteSequence:
        """Generate a continuation of the input sequence.
        
        Args:
            input_sequence: Input sequence to continue
            generator_options: Generator options (ignored in dummy)
            num_steps: Number of steps to generate
            
        Returns:
            Continued sequence
        """
        # Create a copy of the input sequence
        from copy import deepcopy
        output_sequence = deepcopy(input_sequence)
        
        # Find the end time of the input sequence
        if input_sequence.notes:
            end_time = max(note.end_time for note in input_sequence.notes)
        else:
            end_time = 0.0
        
        # Add some random notes
        import random
        
        # Base notes for C major scale
        base_notes = [60, 62, 64, 65, 67, 69, 71, 72]
        
        # Add notes
        current_time = end_time
        for _ in range(num_steps // 4):  # Add fewer notes for simplicity
            note = output_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 0.5
            note.pitch = random.choice(base_notes)
            note.velocity = random.randint(60, 90)
            note.instrument = 0
            note.program = 0
            current_time += 0.25
        
        output_sequence.total_time = current_time
        return output_sequence