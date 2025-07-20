"""Unit tests for model initialization module."""

import pytest
import os
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.music_engine.model_initialization import ModelInitializer


class TestModelInitializer:
    """Test cases for ModelInitializer class."""
    
    @pytest.fixture
    def initializer(self) -> ModelInitializer:
        """Create a ModelInitializer instance for testing."""
        config = SystemConfig(models_path="./test_models")
        return ModelInitializer(config)
    
    def test_initialization(self, initializer: ModelInitializer) -> None:
        """Test initializer initialization."""
        assert initializer is not None
        assert initializer.config is not None
        assert initializer.models_path == "./test_models"
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._verify_model_file")
    def test_ensure_model_available_existing(
        self, mock_verify: MagicMock, mock_makedirs: MagicMock, mock_exists: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test ensuring model availability when model exists."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock verification success
        mock_verify.return_value = True
        
        # Ensure model available
        model_path = initializer.ensure_model_available("performance_rnn")
        
        # Verify behavior
        mock_makedirs.assert_called_once()
        mock_exists.assert_called_once()
        mock_verify.assert_called_once()
        assert "performance_with_dynamics.mag" in model_path
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._download_model")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._verify_model_file")
    def test_ensure_model_available_download(
        self, mock_verify: MagicMock, mock_download: MagicMock, 
        mock_makedirs: MagicMock, mock_exists: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test ensuring model availability when model needs download."""
        # Mock file doesn't exist, then exists after download
        mock_exists.side_effect = [False, True]
        
        # Mock verification success
        mock_verify.return_value = True
        
        # Ensure model available
        model_path = initializer.ensure_model_available("performance_rnn")
        
        # Verify behavior
        mock_makedirs.assert_called_once()
        assert mock_exists.call_count == 2
        mock_download.assert_called_once()
        mock_verify.assert_called_once()
        assert "performance_with_dynamics.mag" in model_path
    
    @patch("os.path.exists")
    def test_ensure_model_available_unknown(
        self, mock_exists: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test ensuring model availability with unknown model type."""
        with pytest.raises(MusicGenerationError) as excinfo:
            initializer.ensure_model_available("unknown_model")
        
        assert "Unknown model type" in str(excinfo.value)
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._download_model")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._verify_model_file")
    def test_ensure_model_available_verification_failure(
        self, mock_verify: MagicMock, mock_download: MagicMock, 
        mock_makedirs: MagicMock, mock_exists: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test ensuring model availability when verification fails."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock verification failure, then success after re-download
        mock_verify.side_effect = [False, True]
        
        # Ensure model available
        model_path = initializer.ensure_model_available("performance_rnn")
        
        # Verify behavior
        mock_makedirs.assert_called_once()
        assert mock_exists.call_count == 1
        mock_download.assert_called_once()
        assert mock_verify.call_count == 2
        assert "performance_with_dynamics.mag" in model_path
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._download_model")
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.ModelInitializer._verify_model_file")
    def test_ensure_model_available_persistent_verification_failure(
        self, mock_verify: MagicMock, mock_download: MagicMock, 
        mock_makedirs: MagicMock, mock_exists: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test ensuring model availability when verification persistently fails."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock verification always fails
        mock_verify.return_value = False
        
        # Ensure model available should fail
        with pytest.raises(MusicGenerationError) as excinfo:
            initializer.ensure_model_available("performance_rnn")
        
        # Verify behavior
        mock_makedirs.assert_called_once()
        assert mock_exists.call_count == 1
        mock_download.assert_called_once()
        assert mock_verify.call_count == 2
        assert "verification failed" in str(excinfo.value)
    
    @patch("requests.get")
    def test_download_model(self, mock_get: MagicMock, initializer: ModelInitializer) -> None:
        """Test model downloading."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1000"
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Mock open file
        mock_file = MagicMock()
        
        # Test with mocked open
        with patch("builtins.open", return_value=mock_file):
            initializer._download_model("http://example.com/model.mag", "model.mag")
        
        # Verify behavior
        mock_get.assert_called_once_with("http://example.com/model.mag", stream=True)
        mock_response.headers.get.assert_called_once_with('content-length', 0)
        mock_response.iter_content.assert_called_once()
        assert mock_file.write.call_count == 2
    
    @patch("requests.get")
    @patch("os.remove")
    def test_download_model_failure(
        self, mock_remove: MagicMock, mock_get: MagicMock, initializer: ModelInitializer
    ) -> None:
        """Test model download failure."""
        # Mock response to raise exception
        mock_get.side_effect = Exception("Download failed")
        
        # Mock path exists
        with patch("os.path.exists", return_value=True):
            # Test download failure
            with pytest.raises(MusicGenerationError) as excinfo:
                initializer._download_model("http://example.com/model.mag", "model.mag")
            
            # Verify behavior
            mock_get.assert_called_once()
            mock_remove.assert_called_once_with("model.mag")
            assert "Failed to download model" in str(excinfo.value)
    
    @patch("os.path.getsize")
    def test_verify_model_file(self, mock_getsize: MagicMock, initializer: ModelInitializer) -> None:
        """Test model file verification."""
        # Mock file size
        mock_getsize.return_value = 10000
        
        # Verify file
        result = initializer._verify_model_file("model.mag")
        
        # Verify behavior
        mock_getsize.assert_called_once_with("model.mag")
        assert result is True
    
    @patch("os.path.getsize")
    def test_verify_model_file_too_small(self, mock_getsize: MagicMock, initializer: ModelInitializer) -> None:
        """Test model file verification with too small file."""
        # Mock file size too small
        mock_getsize.return_value = 500
        
        # Verify file
        result = initializer._verify_model_file("model.mag")
        
        # Verify behavior
        mock_getsize.assert_called_once_with("model.mag")
        assert result is False
    
    @patch("os.path.getsize")
    def test_verify_model_file_error(self, mock_getsize: MagicMock, initializer: ModelInitializer) -> None:
        """Test model file verification with error."""
        # Mock getsize to raise exception
        mock_getsize.side_effect = Exception("File not found")
        
        # Verify file
        result = initializer._verify_model_file("model.mag")
        
        # Verify behavior
        mock_getsize.assert_called_once_with("model.mag")
        assert result is False
    
    def test_get_model_config(self, initializer: ModelInitializer) -> None:
        """Test getting model configuration."""
        # Get config for different model types
        performance_config = initializer.get_model_config("performance_rnn")
        conditional_config = initializer.get_model_config("performance_rnn_conditional")
        melody_config = initializer.get_model_config("melody_rnn")
        
        # Verify configs
        assert performance_config is not None
        assert "temperature" in performance_config
        assert "steps_per_quarter" in performance_config
        
        assert conditional_config is not None
        assert "condition_on_key" in conditional_config
        assert conditional_config["condition_on_key"] is True
        
        assert melody_config is not None
        assert "temperature" in melody_config
    
    def test_get_model_config_unknown(self, initializer: ModelInitializer) -> None:
        """Test getting configuration for unknown model type."""
        with pytest.raises(MusicGenerationError) as excinfo:
            initializer.get_model_config("unknown_model")
        
        assert "Unknown model type" in str(excinfo.value)
    
    def test_create_base_melody(self, initializer: ModelInitializer) -> None:
        """Test base melody creation."""
        # Create melodies of different types
        neutral_melody = initializer.create_base_melody("neutral")
        exciting_melody = initializer.create_base_melody("exciting")
        tense_melody = initializer.create_base_melody("tense")
        sad_melody = initializer.create_base_melody("sad")
        
        # Verify melodies
        assert neutral_melody is not None
        assert len(neutral_melody) > 0
        
        assert exciting_melody is not None
        assert len(exciting_melody) > 0
        
        assert tense_melody is not None
        assert len(tense_melody) > 0
        
        assert sad_melody is not None
        assert len(sad_melody) > 0
    
    def test_create_base_melody_unknown(self, initializer: ModelInitializer) -> None:
        """Test base melody creation with unknown type."""
        # Should default to neutral
        melody = initializer.create_base_melody("unknown_type")
        
        assert melody is not None
        assert len(melody) > 0
    
    def test_create_base_melody_transposed(self, initializer: ModelInitializer) -> None:
        """Test base melody creation with transposition."""
        # Create melody in C
        c_melody = initializer.create_base_melody("neutral", "C")
        
        # Create melody in G (7 semitones up)
        g_melody = initializer.create_base_melody("neutral", "G")
        
        # Verify transposition
        assert len(c_melody) == len(g_melody)
        for i in range(len(c_melody)):
            assert g_melody[i] == c_melody[i] + 7
    
    @patch("crowd_sentiment_music_generator.services.music_engine.model_initialization.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, initializer: ModelInitializer) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(initializer.ensure_model_available, "__wrapped__")
        assert hasattr(initializer.get_model_config, "__wrapped__")
        assert hasattr(initializer.create_base_melody, "__wrapped__")