"""Continuous music evolution module.

This module provides functionality for continuously evolving music in real-time
based on changing emotional parameters and events.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Tuple

import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class ContinuousMusicEvolution:
    """Manages continuous evolution of music in real-time.
    
    This class provides methods for continuously evolving music based on changing
    emotional parameters and events. It runs in a separate thread to ensure smooth
    transitions and continuous music generation.
    """
    
    def __init__(
        self, 
        engine: Optional[MagentaMusicEngine] = None,
        config: Optional[SystemConfig] = None
    ):
        """Initialize the continuous music evolution manager.
        
        Args:
            engine: Magenta music engine (optional, creates a new one if not provided)
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.engine = engine or MagentaMusicEngine(self.config)
        self.is_running = False
        self.evolution_thread = None
        self.target_parameters = None
        self.current_parameters = None
        self.transition_start_time = 0
        self.transition_duration = 0
        self.parameter_queue = []
        self.queue_lock = threading.Lock()
        self.audio_buffer = np.zeros(0)
        self.sample_rate = 44100
        self.audio_callback = None
        self.accent_queue = []
        logger.info("Initialized ContinuousMusicEvolution")
    
    @with_error_handling
    def start(self) -> None:
        """Start the continuous music evolution thread.
        
        Raises:
            MusicGenerationError: If evolution start fails
        """
        if self.is_running:
            logger.warning("Continuous music evolution already running")
            return
        
        try:
            # Initialize the engine if not already initialized
            if not self.engine.is_initialized:
                self.engine.initialize_model()
            
            # Set initial parameters if none exist
            if not self.current_parameters:
                self.current_parameters = MusicalParameters(
                    tempo=100.0,
                    key="C Major",
                    intensity=0.5,
                    instrumentation=["strings", "piano"],
                    mood="neutral",
                    transition_duration=4.0
                )
                self.engine.evolve_music(self.current_parameters)
            
            # Start the evolution thread
            self.is_running = True
            self.evolution_thread = threading.Thread(
                target=self._evolution_loop,
                daemon=True
            )
            self.evolution_thread.start()
            
            logger.info("Started continuous music evolution")
        
        except Exception as e:
            self.is_running = False
            raise MusicGenerationError(f"Failed to start continuous music evolution: {str(e)}")
    
    @with_error_handling
    def stop(self) -> None:
        """Stop the continuous music evolution thread.
        
        Raises:
            MusicGenerationError: If evolution stop fails
        """
        if not self.is_running:
            logger.warning("Continuous music evolution not running")
            return
        
        try:
            # Signal the thread to stop
            self.is_running = False
            
            # Wait for the thread to finish (with timeout)
            if self.evolution_thread and self.evolution_thread.is_alive():
                self.evolution_thread.join(timeout=2.0)
            
            # Clear queues
            with self.queue_lock:
                self.parameter_queue = []
                self.accent_queue = []
            
            logger.info("Stopped continuous music evolution")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to stop continuous music evolution: {str(e)}")
    
    @with_error_handling
    def update_parameters(self, parameters: MusicalParameters) -> None:
        """Update the target musical parameters.
        
        This method queues new parameters for smooth transition.
        
        Args:
            parameters: New target musical parameters
            
        Raises:
            MusicGenerationError: If parameter update fails
        """
        try:
            # Add parameters to the queue
            with self.queue_lock:
                self.parameter_queue.append(parameters)
            
            logger.debug(f"Queued new parameters: tempo={parameters.tempo}, key={parameters.key}, "
                        f"intensity={parameters.intensity}, mood={parameters.mood}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to update parameters: {str(e)}")
    
    @with_error_handling
    def trigger_accent(self, accent_type: str) -> None:
        """Trigger an immediate musical accent.
        
        Args:
            accent_type: Type of accent to trigger (e.g., "goal", "card", "near_miss")
            
        Raises:
            MusicGenerationError: If accent triggering fails
        """
        try:
            # Add accent to the queue
            with self.queue_lock:
                self.accent_queue.append(accent_type)
            
            logger.debug(f"Queued accent: {accent_type}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to trigger accent: {str(e)}")
    
    @with_error_handling
    def set_audio_callback(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """Set a callback function for audio output.
        
        Args:
            callback: Function that accepts audio buffer and sample rate
            
        Raises:
            MusicGenerationError: If callback setting fails
        """
        try:
            self.audio_callback = callback
            logger.debug("Set audio callback")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to set audio callback: {str(e)}")
    
    @with_error_handling
    def get_current_audio(self) -> Tuple[np.ndarray, int]:
        """Get the current audio buffer.
        
        Returns:
            Tuple of (audio_buffer, sample_rate)
            
        Raises:
            MusicGenerationError: If audio retrieval fails
        """
        try:
            return self.audio_buffer, self.sample_rate
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to get current audio: {str(e)}")
    
    def _evolution_loop(self) -> None:
        """Main evolution loop that runs in a separate thread."""
        logger.info("Evolution loop started")
        
        last_update_time = time.time()
        update_interval = self.config.music_update_interval
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for an update
            if current_time - last_update_time >= update_interval:
                try:
                    # Process any queued accents
                    self._process_accents()
                    
                    # Process any queued parameter updates
                    self._process_parameter_updates()
                    
                    # Update audio buffer
                    self._update_audio_buffer()
                    
                    last_update_time = current_time
                
                except Exception as e:
                    logger.error(f"Error in evolution loop: {str(e)}")
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.01)
        
        logger.info("Evolution loop stopped")
    
    def _process_accents(self) -> None:
        """Process any queued accents."""
        with self.queue_lock:
            if not self.accent_queue:
                return
            
            # Get the next accent
            accent_type = self.accent_queue.pop(0)
        
        # Trigger the accent
        try:
            self.engine.trigger_accent(accent_type)
            logger.debug(f"Processed accent: {accent_type}")
        
        except Exception as e:
            logger.error(f"Failed to process accent: {str(e)}")
    
    def _process_parameter_updates(self) -> None:
        """Process any queued parameter updates."""
        with self.queue_lock:
            if not self.parameter_queue:
                return
            
            # Get the next parameters
            parameters = self.parameter_queue.pop(0)
        
        # Check if we're already in a transition
        current_time = time.time()
        in_transition = (self.target_parameters is not None and 
                        current_time < self.transition_start_time + self.transition_duration)
        
        if in_transition:
            # If we're already in a transition, update the target
            self.target_parameters = parameters
            
            # Calculate how far we are into the current transition
            elapsed = current_time - self.transition_start_time
            progress = elapsed / self.transition_duration
            
            if progress < 0.5:
                # If we're less than halfway through, extend the transition
                self.transition_duration = elapsed + parameters.transition_duration
            else:
                # Otherwise, keep the current transition duration
                pass
            
            logger.debug("Updated target parameters during transition")
        
        else:
            # Start a new transition
            self.target_parameters = parameters
            self.transition_start_time = current_time
            self.transition_duration = parameters.transition_duration or 4.0
            
            logger.debug(f"Started new transition to: tempo={parameters.tempo}, key={parameters.key}, "
                        f"intensity={parameters.intensity}, mood={parameters.mood}")
        
        # Calculate interpolated parameters
        self._update_interpolated_parameters()
    
    def _update_interpolated_parameters(self) -> None:
        """Update the current parameters based on transition progress."""
        if not self.target_parameters:
            return
        
        # Calculate progress through transition
        current_time = time.time()
        elapsed = current_time - self.transition_start_time
        
        if elapsed >= self.transition_duration:
            # Transition complete
            self.current_parameters = self.target_parameters
            self.target_parameters = None
            
            # Evolve music with final parameters
            try:
                self.engine.evolve_music(self.current_parameters)
                logger.debug("Transition complete")
            
            except Exception as e:
                logger.error(f"Failed to evolve music at end of transition: {str(e)}")
        
        else:
            # Calculate interpolation factor
            t = elapsed / self.transition_duration
            
            # Create interpolated parameters
            start_params = self.current_parameters
            end_params = self.target_parameters
            
            # Interpolate numeric parameters
            tempo = start_params.tempo + t * (end_params.tempo - start_params.tempo)
            intensity = start_params.intensity + t * (end_params.intensity - start_params.intensity)
            
            # For key, use end key if t > 0.5, otherwise start key
            key = end_params.key if t > 0.5 else start_params.key
            
            # For mood, use end mood if t > 0.5, otherwise start mood
            mood = end_params.mood if t > 0.5 else start_params.mood
            
            # For instrumentation, gradually add end instruments and remove start instruments
            if t < 0.33:
                # Keep start instrumentation
                instrumentation = start_params.instrumentation
            elif t < 0.67:
                # Mix instrumentation
                instrumentation = list(set(start_params.instrumentation + end_params.instrumentation))
            else:
                # Use end instrumentation
                instrumentation = end_params.instrumentation
            
            # Create interpolated parameters
            interpolated = MusicalParameters(
                tempo=tempo,
                key=key,
                intensity=intensity,
                instrumentation=instrumentation,
                mood=mood,
                transition_duration=end_params.transition_duration
            )
            
            # Evolve music with interpolated parameters
            try:
                self.engine.evolve_music(interpolated)
                logger.debug(f"Evolved music with interpolated parameters: {t:.2f} through transition")
            
            except Exception as e:
                logger.error(f"Failed to evolve music during transition: {str(e)}")
    
    def _update_audio_buffer(self) -> None:
        """Update the audio buffer from the engine."""
        try:
            # Get audio from engine
            audio, sample_rate = self.engine.get_audio_output()
            
            # Update buffer
            self.audio_buffer = audio
            self.sample_rate = sample_rate
            
            # Call audio callback if set
            if self.audio_callback:
                self.audio_callback(audio, sample_rate)
        
        except Exception as e:
            logger.error(f"Failed to update audio buffer: {str(e)}")


class TransitionManager:
    """Manages smooth transitions between musical states.
    
    This class provides methods for creating smooth transitions between different
    musical parameters, ensuring continuous and natural evolution of the music.
    """
    
    def __init__(self):
        """Initialize the transition manager."""
        self.transition_curves = {
            "linear": self._linear_transition,
            "exponential": self._exponential_transition,
            "sigmoid": self._sigmoid_transition,
            "ease_in_out": self._ease_in_out_transition
        }
        logger.info("Initialized TransitionManager")
    
    def create_transition(
        self,
        start_params: MusicalParameters,
        end_params: MusicalParameters,
        duration: float,
        steps: int,
        curve_type: str = "ease_in_out"
    ) -> List[MusicalParameters]:
        """Create a smooth transition between musical parameters.
        
        Args:
            start_params: Starting musical parameters
            end_params: Ending musical parameters
            duration: Transition duration in seconds
            steps: Number of steps in the transition
            curve_type: Type of transition curve (linear, exponential, sigmoid, ease_in_out)
            
        Returns:
            List of interpolated parameters for each step
        """
        # Get the appropriate transition curve function
        curve_func = self.transition_curves.get(curve_type, self._ease_in_out_transition)
        
        # Create interpolated parameters for each step
        interpolated_params = []
        
        for i in range(steps):
            # Calculate interpolation factor (0 to 1)
            t = (i + 1) / steps
            
            # Apply curve function to get adjusted factor
            adjusted_t = curve_func(t)
            
            # Create interpolated parameters
            params = self._interpolate_parameters(start_params, end_params, adjusted_t)
            interpolated_params.append(params)
        
        return interpolated_params
    
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
    
    def _linear_transition(self, t: float) -> float:
        """Linear transition curve.
        
        Args:
            t: Input factor (0-1)
            
        Returns:
            Output factor (0-1)
        """
        return t
    
    def _exponential_transition(self, t: float) -> float:
        """Exponential transition curve.
        
        Args:
            t: Input factor (0-1)
            
        Returns:
            Output factor (0-1)
        """
        return t * t
    
    def _sigmoid_transition(self, t: float) -> float:
        """Sigmoid transition curve.
        
        Args:
            t: Input factor (0-1)
            
        Returns:
            Output factor (0-1)
        """
        # Adjusted sigmoid function that maps 0->0 and 1->1
        return 1 / (1 + np.exp(-10 * (t - 0.5)))
    
    def _ease_in_out_transition(self, t: float) -> float:
        """Ease-in-out transition curve.
        
        Args:
            t: Input factor (0-1)
            
        Returns:
            Output factor (0-1)
        """
        # Cubic ease-in-out
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2