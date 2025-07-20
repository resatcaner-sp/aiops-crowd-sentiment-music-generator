"""Unit tests for Magenta music engine module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.models.music.musical_parameters import MusicalParameters
from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine


class TestMagentaMusicEngine:
    """Test cases for MagentaMusicEngine class."""
    
    @pytest.fixture
    def engine(self) -> MagentaMusicEngine:
        """Create a MagentaMusicEngine instance for testing."""
        config = SystemConfig(models_path="./test_models")
        return MagentaMusicEngine(config)
    
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
    
    @pytest.fixture
    def match_context(self) -> dict:
        """Create match context for testing."""
        return {
            "match_importance": 0.8,
            "score_difference": 1,
            "time_remaining": 10,
            "winning_team": "home"
        }
    
    def test_initialization(self, engine: MagentaMusicEngine) -> None:
        """Test engine initialization."""
        assert engine is not None
        assert engine.config is not None
        assert engine.models_path == "./test_models"
        assert not engine.is_initialized
        assert engine.current_sequence is None
    
    @patch("os.path.exists")
    def test_initialize_model(self, mock_exists: MagicMock, engine: MagentaMusicEngine) -> None:
        """Test model initialization."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize model
        engine.initialize_model()
        
        # Verify initialization
        assert engine.is_initialized
        assert engine.generator is not None
        assert hasattr(engine.generator, "generate")
    
    @patch("os.path.exists")
    def test_initialize_model_unknown_type(self, mock_exists: MagicMock, engine: MagentaMusicEngine) -> None:
        """Test initialization with unknown model type."""
        with pytest.raises(MusicGenerationError) as excinfo:
            engine.initialize_model("unknown_model")
        
        assert "Unknown model type" in str(excinfo.value)
    
    @patch("os.path.exists")
    def test_initialize_base_melody(self, mock_exists: MagicMock, engine: MagentaMusicEngine, match_context: dict) -> None:
        """Test base melody initialization."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize base melody
        engine.initialize_base_melody(match_context)
        
        # Verify initialization
        assert engine.current_sequence is not None
        assert len(engine.current_sequence.notes) > 0
        assert engine.generation_start_time > 0
    
    @patch("os.path.exists")
    def test_evolve_music(
        self, mock_exists: MagicMock, engine: MagentaMusicEngine, musical_parameters: MusicalParameters
    ) -> None:
        """Test music evolution."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize model
        engine.initialize_model()
        
        # Evolve music
        engine.evolve_music(musical_parameters)
        
        # Verify evolution
        assert engine.current_sequence is not None
        assert engine.current_parameters == musical_parameters
    
    @patch("os.path.exists")
    def test_trigger_accent(self, mock_exists: MagicMock, engine: MagentaMusicEngine) -> None:
        """Test accent triggering."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize model and base melody
        engine.initialize_model()
        engine.initialize_base_melody()
        
        # Get initial sequence length
        initial_notes = len(engine.current_sequence.notes)
        
        # Trigger accent
        engine.trigger_accent("goal")
        
        # Verify accent was added
        assert len(engine.current_sequence.notes) > initial_notes
    
    @patch("os.path.exists")
    def test_transition_to(
        self, mock_exists: MagicMock, engine: MagentaMusicEngine, musical_parameters: MusicalParameters
    ) -> None:
        """Test transition to new parameters."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize model and set current parameters
        engine.initialize_model()
        engine.current_parameters = MusicalParameters(
            tempo=100.0,
            key="A Minor",
            intensity=0.4,
            instrumentation=["piano"],
            mood="somber",
            transition_duration=2.0
        )
        
        # Transition to new parameters
        engine.transition_to(musical_parameters, 2.0)
        
        # Verify transition
        assert engine.current_parameters is not None
        assert engine.current_parameters.tempo == musical_parameters.tempo
        assert engine.current_parameters.key == musical_parameters.key
        assert engine.current_parameters.intensity == musical_parameters.intensity
    
    @patch("os.path.exists")
    def test_get_audio_output(self, mock_exists: MagicMock, engine: MagentaMusicEngine, musical_parameters: MusicalParameters) -> None:
        """Test audio output generation."""
        # Mock os.path.exists to return False so we use the dummy generator
        mock_exists.return_value = False
        
        # Initialize model and set parameters
        engine.initialize_model()
        engine.current_parameters = musical_parameters
        engine.current_sequence = MagicMock()
        
        # Get audio output
        audio, sample_rate = engine.get_audio_output()
        
        # Verify output
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sample_rate == engine.sample_rate
    
    def test_map_intensity_to_temperature(self, engine: MagentaMusicEngine) -> None:
        """Test mapping intensity to temperature."""
        # Test various intensity values
        high_intensity = engine._map_intensity_to_temperature(0.9)
        medium_intensity = engine._map_intensity_to_temperature(0.5)
        low_intensity = engine._map_intensity_to_temperature(0.1)
        
        # Verify mapping (higher intensity = lower temperature)
        assert high_intensity < medium_intensity < low_intensity
    
    def test_interpolate_parameters(self, engine: MagentaMusicEngine) -> None:
        """Test parameter interpolation."""
        # Create start and end parameters
        start_params = MusicalParameters(
            tempo=100.0,
            key="C Major",
            intensity=0.4,
            instrumentation=["piano"],
            mood="neutral",
            transition_duration=5.0
        )
        
        end_params = MusicalParameters(
            tempo=140.0,
            key="G Major",
            intensity=0.8,
            instrumentation=["strings", "brass"],
            mood="energetic",
            transition_duration=2.0
        )
        
        # Test interpolation at different points
        t25 = engine._interpolate_parameters(start_params, end_params, 0.25)
        t50 = engine._interpolate_parameters(start_params, end_params, 0.5)
        t75 = engine._interpolate_parameters(start_params, end_params, 0.75)
        
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
    
    def test_create_goal_accent(self, engine: MagentaMusicEngine) -> None:
        """Test goal accent creation."""
        accent = engine._create_goal_accent()
        assert accent is not None
        assert len(accent.notes) > 0
        assert any(note.program == 61 for note in accent.notes)  # Brass
    
    def test_create_card_accent(self, engine: MagentaMusicEngine) -> None:
        """Test card accent creation."""
        accent = engine._create_card_accent()
        assert accent is not None
        assert len(accent.notes) > 0
        assert all(note.start_time == 0.0 for note in accent.notes)  # Simultaneous chord
    
    def test_create_near_miss_accent(self, engine: MagentaMusicEngine) -> None:
        """Test near miss accent creation."""
        accent = engine._create_near_miss_accent()
        assert accent is not None
        assert len(accent.notes) > 0
        
        # Check for rising pattern
        pitches = [note.pitch for note in sorted(accent.notes, key=lambda n: n.start_time)]
        assert all(pitches[i] < pitches[i+1] for i in range(len(pitches)-1))
    
    @patch("crowd_sentiment_music_generator.services.music_engine.magenta_engine.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, engine: MagentaMusicEngine) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(engine.initialize_model, "__wrapped__")
        assert hasattr(engine.initialize_base_melody, "__wrapped__")
        assert hasattr(engine.evolve_music, "__wrapped__")
        assert hasattr(engine.trigger_accent, "__wrapped__")
        assert hasattr(engine.transition_to, "__wrapped__")
        assert hasattr(engine.get_audio_output, "__wrapped__")