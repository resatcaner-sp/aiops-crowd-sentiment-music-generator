"""Unit tests for accent triggering module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.music_engine.accent_triggering import AccentTrigger


class TestAccentTrigger:
    """Test cases for AccentTrigger class."""
    
    @pytest.fixture
    def trigger(self) -> AccentTrigger:
        """Create an AccentTrigger instance for testing."""
        config = SystemConfig()
        return AccentTrigger(config)
    
    def test_initialization(self, trigger: AccentTrigger) -> None:
        """Test trigger initialization."""
        assert trigger is not None
        assert trigger.config is not None
        assert trigger.sample_rate == 44100
    
    def test_create_accent_sequence_known_type(self, trigger: AccentTrigger) -> None:
        """Test creating an accent sequence for a known type."""
        # Create accent sequences for different types
        goal_accent = trigger.create_accent_sequence("goal")
        card_accent = trigger.create_accent_sequence("card")
        near_miss_accent = trigger.create_accent_sequence("near_miss")
        penalty_accent = trigger.create_accent_sequence("penalty")
        corner_accent = trigger.create_accent_sequence("corner")
        
        # Verify sequences
        assert goal_accent is not None
        assert len(goal_accent.notes) > 0
        assert any(note.program == 61 for note in goal_accent.notes)  # Brass
        
        assert card_accent is not None
        assert len(card_accent.notes) > 0
        assert all(note.start_time == 0.0 for note in card_accent.notes)  # Simultaneous chord
        
        assert near_miss_accent is not None
        assert len(near_miss_accent.notes) > 0
        
        assert penalty_accent is not None
        assert len(penalty_accent.notes) > 0
        
        assert corner_accent is not None
        assert len(corner_accent.notes) > 0
    
    def test_create_accent_sequence_unknown_type(self, trigger: AccentTrigger) -> None:
        """Test creating an accent sequence for an unknown type."""
        # Create accent sequence for unknown type
        accent = trigger.create_accent_sequence("unknown_type")
        
        # Verify default sequence was created
        assert accent is not None
        assert len(accent.notes) > 0
        assert all(note.program == 0 for note in accent.notes)  # Piano
    
    def test_create_accent_audio_known_type(self, trigger: AccentTrigger) -> None:
        """Test creating accent audio for a known type."""
        # Create accent audio for different types
        goal_audio, goal_rate = trigger.create_accent_audio("goal")
        card_audio, card_rate = trigger.create_accent_audio("card")
        near_miss_audio, near_miss_rate = trigger.create_accent_audio("near_miss")
        
        # Verify audio
        assert isinstance(goal_audio, np.ndarray)
        assert len(goal_audio) > 0
        assert goal_rate == trigger.sample_rate
        
        assert isinstance(card_audio, np.ndarray)
        assert len(card_audio) > 0
        assert card_rate == trigger.sample_rate
        
        assert isinstance(near_miss_audio, np.ndarray)
        assert len(near_miss_audio) > 0
        assert near_miss_rate == trigger.sample_rate
    
    def test_create_accent_audio_unknown_type(self, trigger: AccentTrigger) -> None:
        """Test creating accent audio for an unknown type."""
        # Create accent audio for unknown type
        audio, rate = trigger.create_accent_audio("unknown_type")
        
        # Verify default audio was created
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert rate == trigger.sample_rate
    
    def test_get_accent_priority(self, trigger: AccentTrigger) -> None:
        """Test getting accent priority."""
        # Get priorities for different types
        goal_priority = trigger.get_accent_priority("goal")
        card_priority = trigger.get_accent_priority("card")
        corner_priority = trigger.get_accent_priority("corner")
        unknown_priority = trigger.get_accent_priority("unknown_type")
        
        # Verify priorities
        assert goal_priority == 10
        assert card_priority == 7
        assert corner_priority == 3
        assert unknown_priority == 1  # Default priority for unknown types
    
    def test_fanfare_accent(self, trigger: AccentTrigger) -> None:
        """Test creating a fanfare accent."""
        # Create fanfare accent
        definition = {
            "intensity": 1.0,
            "duration": 2.0,
            "instruments": ["brass", "percussion"],
            "pattern": "fanfare",
            "priority": 10
        }
        accent = trigger._create_fanfare_accent(definition)
        
        # Verify accent
        assert accent is not None
        assert len(accent.notes) > 0
        assert any(note.program == 61 for note in accent.notes)  # Brass
        assert any(note.program == 128 for note in accent.notes)  # Percussion
    
    def test_staccato_accent(self, trigger: AccentTrigger) -> None:
        """Test creating a staccato accent."""
        # Create staccato accent
        definition = {
            "intensity": 0.7,
            "duration": 1.0,
            "instruments": ["strings", "percussion"],
            "pattern": "staccato",
            "priority": 7
        }
        accent = trigger._create_staccato_accent(definition)
        
        # Verify accent
        assert accent is not None
        assert len(accent.notes) > 0
        assert all(note.start_time == 0.0 for note in accent.notes)  # Simultaneous chord
        assert all(note.program == 48 for note in accent.notes)  # Strings
    
    def test_rising_accent(self, trigger: AccentTrigger) -> None:
        """Test creating a rising accent."""
        # Create rising accent
        definition = {
            "intensity": 0.6,
            "duration": 1.5,
            "instruments": ["strings", "woodwinds"],
            "pattern": "rising",
            "priority": 5
        }
        accent = trigger._create_rising_accent(definition)
        
        # Verify accent
        assert accent is not None
        assert len(accent.notes) > 0
        
        # Check for rising pattern
        pitches = [note.pitch for note in sorted(accent.notes, key=lambda n: n.start_time)]
        assert all(pitches[i] < pitches[i+1] for i in range(len(pitches)-1))
    
    def test_tension_accent(self, trigger: AccentTrigger) -> None:
        """Test creating a tension accent."""
        # Create tension accent
        definition = {
            "intensity": 0.8,
            "duration": 1.5,
            "instruments": ["percussion", "brass"],
            "pattern": "tension",
            "priority": 8
        }
        accent = trigger._create_tension_accent(definition)
        
        # Verify accent
        assert accent is not None
        assert len(accent.notes) > 0
        assert all(note.program == 128 for note in accent.notes)  # Percussion
    
    def test_default_accent(self, trigger: AccentTrigger) -> None:
        """Test creating a default accent."""
        # Create default accent
        accent = trigger._create_default_accent()
        
        # Verify accent
        assert accent is not None
        assert len(accent.notes) > 0
        assert all(note.program == 0 for note in accent.notes)  # Piano
    
    def test_fanfare_audio(self, trigger: AccentTrigger) -> None:
        """Test creating fanfare audio."""
        # Create fanfare audio
        audio, rate = trigger._create_fanfare_audio(0.8, 2.0)
        
        # Verify audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(2.0 * trigger.sample_rate)
        assert rate == trigger.sample_rate
        assert np.max(np.abs(audio)) <= 0.8  # Intensity scaling
    
    def test_staccato_audio(self, trigger: AccentTrigger) -> None:
        """Test creating staccato audio."""
        # Create staccato audio
        audio, rate = trigger._create_staccato_audio(0.7, 1.0)
        
        # Verify audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(1.0 * trigger.sample_rate)
        assert rate == trigger.sample_rate
        assert np.max(np.abs(audio)) <= 0.7  # Intensity scaling
    
    def test_rising_audio(self, trigger: AccentTrigger) -> None:
        """Test creating rising audio."""
        # Create rising audio
        audio, rate = trigger._create_rising_audio(0.6, 1.5)
        
        # Verify audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(1.5 * trigger.sample_rate)
        assert rate == trigger.sample_rate
        assert np.max(np.abs(audio)) <= 0.6  # Intensity scaling
    
    def test_tension_audio(self, trigger: AccentTrigger) -> None:
        """Test creating tension audio."""
        # Create tension audio
        audio, rate = trigger._create_tension_audio(0.8, 1.5)
        
        # Verify audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(1.5 * trigger.sample_rate)
        assert rate == trigger.sample_rate
        assert np.max(np.abs(audio)) <= 0.8  # Intensity scaling
    
    def test_default_audio(self, trigger: AccentTrigger) -> None:
        """Test creating default audio."""
        # Create default audio
        audio, rate = trigger._create_default_audio()
        
        # Verify audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert rate == trigger.sample_rate
        assert np.max(np.abs(audio)) <= 0.5  # Default intensity
    
    @patch("crowd_sentiment_music_generator.services.music_engine.accent_triggering.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, trigger: AccentTrigger) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(trigger.create_accent_sequence, "__wrapped__")
        assert hasattr(trigger.create_accent_audio, "__wrapped__")
        assert hasattr(trigger.get_accent_priority, "__wrapped__")