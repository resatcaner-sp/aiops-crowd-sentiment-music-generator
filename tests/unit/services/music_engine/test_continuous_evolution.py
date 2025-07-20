"""Unit tests for continuous music evolution module."""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch
import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_engine.continuous_evolution import ContinuousMusicEvolution, TransitionManager


class TestContinuousMusicEvolution:
    """Test cases for ContinuousMusicEvolution class."""
    
    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock Magenta music engine."""
        engine = MagicMock()
        engine.is_initialized = False
        engine.get_audio_output.return_value = (np.zeros(1000), 44100)
        return engine
    
    @pytest.fixture
    def evolution(self, mock_engine: MagicMock) -> ContinuousMusicEvolution:
        """Create a ContinuousMusicEvolution instance for testing."""
        config = SystemConfig(music_update_interval=0.01)  # Fast updates for testing
        return ContinuousMusicEvolution(engine=mock_engine, config=config)
    
    @pytest.fixture
    def musical_parameters(self) -> MusicalParameters:
        """Create musical parameters for testing."""
        return MusicalParameters(
            tempo=120.0,
            key="C Major",
            intensity=0.7,
            instrumentation=["piano", "strings"],
            mood="energetic",
            transition_duration=3.0
        )
    
    def test_initialization(self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock) -> None:
        """Test evolution initialization."""
        assert evolution is not None
        assert evolution.config is not None
        assert evolution.engine == mock_engine
        assert not evolution.is_running
        assert evolution.evolution_thread is None
        assert evolution.target_parameters is None
        assert evolution.current_parameters is None
    
    def test_start_stop(self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock) -> None:
        """Test starting and stopping the evolution thread."""
        # Start evolution
        evolution.start()
        
        # Verify state
        assert evolution.is_running
        assert evolution.evolution_thread is not None
        assert evolution.evolution_thread.is_alive()
        mock_engine.initialize_model.assert_called_once()
        
        # Stop evolution
        evolution.stop()
        
        # Verify state
        assert not evolution.is_running
        
        # Wait for thread to actually stop
        time.sleep(0.1)
        assert not evolution.evolution_thread.is_alive()
    
    def test_update_parameters(
        self, evolution: ContinuousMusicEvolution, musical_parameters: MusicalParameters
    ) -> None:
        """Test updating parameters."""
        # Update parameters
        evolution.update_parameters(musical_parameters)
        
        # Verify parameters were queued
        with evolution.queue_lock:
            assert len(evolution.parameter_queue) == 1
            assert evolution.parameter_queue[0] == musical_parameters
    
    def test_trigger_accent(self, evolution: ContinuousMusicEvolution) -> None:
        """Test triggering an accent."""
        # Trigger accent
        evolution.trigger_accent("goal")
        
        # Verify accent was queued
        with evolution.queue_lock:
            assert len(evolution.accent_queue) == 1
            assert evolution.accent_queue[0] == "goal"
    
    def test_set_audio_callback(self, evolution: ContinuousMusicEvolution) -> None:
        """Test setting the audio callback."""
        # Create mock callback
        callback = MagicMock()
        
        # Set callback
        evolution.set_audio_callback(callback)
        
        # Verify callback was set
        assert evolution.audio_callback == callback
    
    def test_get_current_audio(self, evolution: ContinuousMusicEvolution) -> None:
        """Test getting the current audio buffer."""
        # Set test buffer
        test_buffer = np.ones(1000)
        evolution.audio_buffer = test_buffer
        evolution.sample_rate = 48000
        
        # Get audio
        audio, sample_rate = evolution.get_current_audio()
        
        # Verify audio
        assert np.array_equal(audio, test_buffer)
        assert sample_rate == 48000
    
    @patch("time.sleep")
    def test_evolution_loop(
        self, mock_sleep: MagicMock, evolution: ContinuousMusicEvolution, 
        mock_engine: MagicMock, musical_parameters: MusicalParameters
    ) -> None:
        """Test the evolution loop."""
        # Set up initial state
        evolution.current_parameters = musical_parameters
        evolution.is_running = True
        
        # Mock time.time to return increasing values
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.1, 0.2, 0.3]
            
            # Run a few iterations of the loop
            for _ in range(3):
                evolution._evolution_loop()
                
                # Break after a few iterations
                if _ >= 2:
                    evolution.is_running = False
        
        # Verify engine was called to update audio
        assert mock_engine.get_audio_output.call_count > 0
    
    def test_process_accents(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock
    ) -> None:
        """Test processing accents."""
        # Add accents to queue
        with evolution.queue_lock:
            evolution.accent_queue.append("goal")
            evolution.accent_queue.append("card")
        
        # Process accents
        evolution._process_accents()
        
        # Verify accent was processed
        mock_engine.trigger_accent.assert_called_once_with("goal")
        
        # Verify queue state
        with evolution.queue_lock:
            assert len(evolution.accent_queue) == 1
            assert evolution.accent_queue[0] == "card"
    
    def test_process_parameter_updates_new_transition(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock, musical_parameters: MusicalParameters
    ) -> None:
        """Test processing parameter updates with a new transition."""
        # Set up initial state
        evolution.current_parameters = MusicalParameters(
            tempo=100.0,
            key="A Minor",
            intensity=0.4,
            instrumentation=["piano"],
            mood="somber",
            transition_duration=2.0
        )
        
        # Add parameters to queue
        with evolution.queue_lock:
            evolution.parameter_queue.append(musical_parameters)
        
        # Mock time.time
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            
            # Process parameter updates
            evolution._process_parameter_updates()
        
        # Verify transition state
        assert evolution.target_parameters == musical_parameters
        assert evolution.transition_start_time == 1000.0
        assert evolution.transition_duration == 3.0
        
        # Verify queue state
        with evolution.queue_lock:
            assert len(evolution.parameter_queue) == 0
    
    def test_process_parameter_updates_during_transition(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock, musical_parameters: MusicalParameters
    ) -> None:
        """Test processing parameter updates during an existing transition."""
        # Set up initial state
        evolution.current_parameters = MusicalParameters(
            tempo=100.0,
            key="A Minor",
            intensity=0.4,
            instrumentation=["piano"],
            mood="somber",
            transition_duration=2.0
        )
        evolution.target_parameters = MusicalParameters(
            tempo=110.0,
            key="C Major",
            intensity=0.5,
            instrumentation=["strings"],
            mood="neutral",
            transition_duration=4.0
        )
        evolution.transition_start_time = 1000.0
        evolution.transition_duration = 4.0
        
        # Add parameters to queue
        with evolution.queue_lock:
            evolution.parameter_queue.append(musical_parameters)
        
        # Mock time.time to return a time during the transition
        with patch("time.time") as mock_time:
            mock_time.return_value = 1001.0  # 1 second into transition
            
            # Process parameter updates
            evolution._process_parameter_updates()
        
        # Verify transition state
        assert evolution.target_parameters == musical_parameters
        assert evolution.transition_start_time == 1000.0
        assert evolution.transition_duration == 5.0  # Extended by 1 + 4
        
        # Verify queue state
        with evolution.queue_lock:
            assert len(evolution.parameter_queue) == 0
    
    def test_update_interpolated_parameters_complete(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock, musical_parameters: MusicalParameters
    ) -> None:
        """Test updating interpolated parameters when transition is complete."""
        # Set up initial state
        evolution.current_parameters = MusicalParameters(
            tempo=100.0,
            key="A Minor",
            intensity=0.4,
            instrumentation=["piano"],
            mood="somber",
            transition_duration=2.0
        )
        evolution.target_parameters = musical_parameters
        evolution.transition_start_time = 1000.0
        evolution.transition_duration = 3.0
        
        # Mock time.time to return a time after the transition
        with patch("time.time") as mock_time:
            mock_time.return_value = 1004.0  # 4 seconds after start (transition is 3 seconds)
            
            # Update interpolated parameters
            evolution._update_interpolated_parameters()
        
        # Verify state
        assert evolution.current_parameters == musical_parameters
        assert evolution.target_parameters is None
        mock_engine.evolve_music.assert_called_once_with(musical_parameters)
    
    def test_update_interpolated_parameters_in_progress(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock, musical_parameters: MusicalParameters
    ) -> None:
        """Test updating interpolated parameters during transition."""
        # Set up initial state
        start_params = MusicalParameters(
            tempo=100.0,
            key="A Minor",
            intensity=0.4,
            instrumentation=["piano"],
            mood="somber",
            transition_duration=2.0
        )
        evolution.current_parameters = start_params
        evolution.target_parameters = musical_parameters
        evolution.transition_start_time = 1000.0
        evolution.transition_duration = 4.0
        
        # Mock time.time to return a time during the transition
        with patch("time.time") as mock_time:
            mock_time.return_value = 1002.0  # 2 seconds into transition (50%)
            
            # Update interpolated parameters
            evolution._update_interpolated_parameters()
        
        # Verify engine was called with interpolated parameters
        mock_engine.evolve_music.assert_called_once()
        call_args = mock_engine.evolve_music.call_args[0][0]
        
        # Check interpolated values
        assert call_args.tempo == 110.0  # Halfway between 100 and 120
        assert call_args.intensity == 0.55  # Halfway between 0.4 and 0.7
        assert call_args.key == musical_parameters.key  # t > 0.5, use target key
        assert call_args.mood == musical_parameters.mood  # t > 0.5, use target mood
    
    def test_update_audio_buffer(
        self, evolution: ContinuousMusicEvolution, mock_engine: MagicMock
    ) -> None:
        """Test updating the audio buffer."""
        # Set up mock engine to return test audio
        test_audio = np.ones(1000)
        test_sample_rate = 48000
        mock_engine.get_audio_output.return_value = (test_audio, test_sample_rate)
        
        # Set up mock callback
        callback = MagicMock()
        evolution.audio_callback = callback
        
        # Update audio buffer
        evolution._update_audio_buffer()
        
        # Verify buffer was updated
        assert np.array_equal(evolution.audio_buffer, test_audio)
        assert evolution.sample_rate == test_sample_rate
        
        # Verify callback was called
        callback.assert_called_once_with(test_audio, test_sample_rate)
    
    @patch("crowd_sentiment_music_generator.services.music_engine.continuous_evolution.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, evolution: ContinuousMusicEvolution) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(evolution.start, "__wrapped__")
        assert hasattr(evolution.stop, "__wrapped__")
        assert hasattr(evolution.update_parameters, "__wrapped__")
        assert hasattr(evolution.trigger_accent, "__wrapped__")
        assert hasattr(evolution.set_audio_callback, "__wrapped__")
        assert hasattr(evolution.get_current_audio, "__wrapped__")


class TestTransitionManager:
    """Test cases for TransitionManager class."""
    
    @pytest.fixture
    def manager(self) -> TransitionManager:
        """Create a TransitionManager instance for testing."""
        return TransitionManager()
    
    @pytest.fixture
    def start_params(self) -> MusicalParameters:
        """Create starting parameters for testing."""
        return MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.4,
            instrumentation=["piano"],
            mood="neutral",
            transition_duration=5.0
        )
    
    @pytest.fixture
    def end_params(self) -> MusicalParameters:
        """Create ending parameters for testing."""
        return MusicalParameters(
            tempo=140.0,
            key="G Major",
            intensity=0.8,
            instrumentation=["strings", "brass"],
            mood="energetic",
            transition_duration=2.0
        )
    
    def test_initialization(self, manager: TransitionManager) -> None:
        """Test manager initialization."""
        assert manager is not None
        assert "linear" in manager.transition_curves
        assert "exponential" in manager.transition_curves
        assert "sigmoid" in manager.transition_curves
        assert "ease_in_out" in manager.transition_curves
    
    def test_create_transition_linear(
        self, manager: TransitionManager, start_params: MusicalParameters, end_params: MusicalParameters
    ) -> None:
        """Test creating a linear transition."""
        # Create transition
        steps = 5
        params_list = manager.create_transition(
            start_params, end_params, 2.0, steps, "linear"
        )
        
        # Verify transition
        assert len(params_list) == steps
        
        # Check first step
        assert params_list[0].tempo == 108.0  # 100 + 0.2 * (140 - 100)
        assert params_list[0].intensity == 0.48  # 0.4 + 0.2 * (0.8 - 0.4)
        
        # Check last step
        assert params_list[-1].tempo == 140.0  # 100 + 1.0 * (140 - 100)
        assert params_list[-1].intensity == 0.8  # 0.4 + 1.0 * (0.8 - 0.4)
    
    def test_create_transition_exponential(
        self, manager: TransitionManager, start_params: MusicalParameters, end_params: MusicalParameters
    ) -> None:
        """Test creating an exponential transition."""
        # Create transition
        steps = 5
        params_list = manager.create_transition(
            start_params, end_params, 2.0, steps, "exponential"
        )
        
        # Verify transition
        assert len(params_list) == steps
        
        # Check first step (t=0.2, t^2=0.04)
        first_t = 0.2
        first_t_squared = first_t * first_t
        expected_first_tempo = start_params.tempo + first_t_squared * (end_params.tempo - start_params.tempo)
        assert abs(params_list[0].tempo - expected_first_tempo) < 0.001
    
    def test_create_transition_sigmoid(
        self, manager: TransitionManager, start_params: MusicalParameters, end_params: MusicalParameters
    ) -> None:
        """Test creating a sigmoid transition."""
        # Create transition
        steps = 5
        params_list = manager.create_transition(
            start_params, end_params, 2.0, steps, "sigmoid"
        )
        
        # Verify transition
        assert len(params_list) == steps
        
        # Check that middle step has more change than first step
        first_step_tempo_change = params_list[0].tempo - start_params.tempo
        middle_step_tempo_change = params_list[2].tempo - params_list[1].tempo
        assert middle_step_tempo_change > first_step_tempo_change
    
    def test_create_transition_ease_in_out(
        self, manager: TransitionManager, start_params: MusicalParameters, end_params: MusicalParameters
    ) -> None:
        """Test creating an ease-in-out transition."""
        # Create transition
        steps = 5
        params_list = manager.create_transition(
            start_params, end_params, 2.0, steps, "ease_in_out"
        )
        
        # Verify transition
        assert len(params_list) == steps
        
        # Check that middle step has more change than first step
        first_step_tempo_change = params_list[0].tempo - start_params.tempo
        middle_step_tempo_change = params_list[2].tempo - params_list[1].tempo
        assert middle_step_tempo_change > first_step_tempo_change
    
    def test_interpolate_parameters(
        self, manager: TransitionManager, start_params: MusicalParameters, end_params: MusicalParameters
    ) -> None:
        """Test parameter interpolation."""
        # Test interpolation at different points
        t25 = manager._interpolate_parameters(start_params, end_params, 0.25)
        t50 = manager._interpolate_parameters(start_params, end_params, 0.5)
        t75 = manager._interpolate_parameters(start_params, end_params, 0.75)
        
        # Verify interpolation
        assert t25.tempo == 110.0  # 100 + 0.25 * (140 - 100)
        assert t50.tempo == 120.0  # 100 + 0.5 * (140 - 100)
        assert t75.tempo == 130.0  # 100 + 0.75 * (140 - 100)
        
        assert t25.key == start_params.key  # t < 0.5, use start key
        assert t50.key == end_params.key  # t >= 0.5, use end key
        assert t75.key == end_params.key  # t >= 0.5, use end key
        
        assert t25.mood == start_params.mood  # t < 0.5, use start mood
        assert t50.mood == end_params.mood  # t >= 0.5, use end mood
        assert t75.mood == end_params.mood  # t >= 0.5, use end mood
        
        # Instrumentation changes in steps
        assert set(t25.instrumentation) == set(start_params.instrumentation)
        assert set(t50.instrumentation) == set(start_params.instrumentation + end_params.instrumentation)
        assert set(t75.instrumentation) == set(end_params.instrumentation)
    
    def test_transition_curves(self, manager: TransitionManager) -> None:
        """Test transition curve functions."""
        # Test linear curve
        assert manager._linear_transition(0.0) == 0.0
        assert manager._linear_transition(0.5) == 0.5
        assert manager._linear_transition(1.0) == 1.0
        
        # Test exponential curve
        assert manager._exponential_transition(0.0) == 0.0
        assert manager._exponential_transition(0.5) == 0.25
        assert manager._exponential_transition(1.0) == 1.0
        
        # Test sigmoid curve
        assert manager._sigmoid_transition(0.0) < 0.1
        assert abs(manager._sigmoid_transition(0.5) - 0.5) < 0.1
        assert manager._sigmoid_transition(1.0) > 0.9
        
        # Test ease-in-out curve
        assert manager._ease_in_out_transition(0.0) == 0.0
        assert abs(manager._ease_in_out_transition(0.5) - 0.5) < 0.1
        assert manager._ease_in_out_transition(1.0) == 1.0