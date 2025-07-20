"""Unit tests for audio pipeline module."""

import pytest
import os
import io
from unittest.mock import MagicMock, patch
import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.music_engine.audio_pipeline import AudioOutputPipeline


class TestAudioOutputPipeline:
    """Test cases for AudioOutputPipeline class."""
    
    @pytest.fixture
    def pipeline(self) -> AudioOutputPipeline:
        """Create an AudioOutputPipeline instance for testing."""
        config = SystemConfig(buffer_size=1)  # 1 second buffer for testing
        return AudioOutputPipeline(config)
    
    @pytest.fixture
    def mono_frame(self) -> np.ndarray:
        """Create a mono audio frame for testing."""
        # 0.5 seconds of 440 Hz sine wave at 44100 Hz
        t = np.linspace(0, 0.5, int(44100 * 0.5), False)
        return np.sin(2 * np.pi * 440 * t)
    
    @pytest.fixture
    def stereo_frame(self) -> np.ndarray:
        """Create a stereo audio frame for testing."""
        # 0.5 seconds of 440 Hz sine wave at 44100 Hz
        t = np.linspace(0, 0.5, int(44100 * 0.5), False)
        mono = np.sin(2 * np.pi * 440 * t)
        return np.column_stack((mono, mono))
    
    def test_initialization(self, pipeline: AudioOutputPipeline) -> None:
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.sample_rate == 44100
        assert pipeline.channels == 2
        assert pipeline.buffer_size == 44100  # 1 second at 44100 Hz
        assert pipeline.audio_buffer.shape == (44100, 2)
        assert pipeline.buffer_position == 0
        assert not pipeline.is_recording
    
    def test_process_audio_frame_mono(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test processing a mono audio frame."""
        # Process frame
        pipeline.process_audio_frame(mono_frame, 44100)
        
        # Verify buffer state
        assert pipeline.buffer_position == len(mono_frame)
        
        # Check that mono was converted to stereo
        buffer_slice = pipeline.audio_buffer[:len(mono_frame)]
        assert buffer_slice.shape[1] == 2
        assert np.array_equal(buffer_slice[:, 0], buffer_slice[:, 1])
    
    def test_process_audio_frame_stereo(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test processing a stereo audio frame."""
        # Process frame
        pipeline.process_audio_frame(stereo_frame, 44100)
        
        # Verify buffer state
        assert pipeline.buffer_position == len(stereo_frame)
        
        # Check that stereo was preserved
        buffer_slice = pipeline.audio_buffer[:len(stereo_frame)]
        assert buffer_slice.shape[1] == 2
        assert np.array_equal(buffer_slice, stereo_frame)
    
    def test_process_audio_frame_resampling(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test processing a frame with different sample rate."""
        # Mock resampling function
        with patch.object(pipeline, "_resample_audio") as mock_resample:
            # Set up mock to return the original frame
            mock_resample.return_value = mono_frame
            
            # Process frame with different sample rate
            pipeline.process_audio_frame(mono_frame, 22050)
            
            # Verify resampling was called
            mock_resample.assert_called_once_with(mono_frame, 22050, 44100)
    
    def test_process_audio_frame_wrapping(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test buffer wrapping when processing frames."""
        # Set buffer position near the end
        pipeline.buffer_position = pipeline.buffer_size - len(mono_frame) // 2
        
        # Process frame
        pipeline.process_audio_frame(mono_frame, 44100)
        
        # Verify buffer position wrapped around
        assert pipeline.buffer_position == len(mono_frame) // 2
    
    def test_process_audio_frame_callback(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test output callback when processing frames."""
        # Set up mock callback
        callback = MagicMock()
        pipeline.set_output_callback(callback)
        
        # Process frame
        pipeline.process_audio_frame(mono_frame, 44100)
        
        # Verify callback was called
        callback.assert_called_once()
        args = callback.call_args[0]
        assert len(args) == 2
        assert isinstance(args[0], np.ndarray)
        assert args[1] == 44100
    
    def test_set_output_callback(self, pipeline: AudioOutputPipeline) -> None:
        """Test setting the output callback."""
        # Set up mock callback
        callback = MagicMock()
        
        # Set callback
        pipeline.set_output_callback(callback)
        
        # Verify callback was set
        assert pipeline.output_callback == callback
    
    def test_get_current_buffer(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test getting the current buffer."""
        # Process a frame
        pipeline.process_audio_frame(stereo_frame, 44100)
        
        # Get buffer
        buffer, sample_rate = pipeline.get_current_buffer()
        
        # Verify buffer
        assert isinstance(buffer, np.ndarray)
        assert buffer.shape == pipeline.audio_buffer.shape
        assert sample_rate == pipeline.sample_rate
    
    def test_start_recording(self, pipeline: AudioOutputPipeline) -> None:
        """Test starting recording."""
        # Start recording
        pipeline.start_recording()
        
        # Verify recording state
        assert pipeline.is_recording
        assert pipeline.recording_buffer.shape == (0, 2)
    
    def test_start_recording_already_recording(self, pipeline: AudioOutputPipeline) -> None:
        """Test starting recording when already recording."""
        # Start recording
        pipeline.start_recording()
        
        # Start recording again
        pipeline.start_recording()
        
        # Verify recording state
        assert pipeline.is_recording
    
    def test_stop_recording(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test stopping recording."""
        # Start recording
        pipeline.start_recording()
        
        # Process a frame
        pipeline.process_audio_frame(stereo_frame, 44100)
        
        # Stop recording
        recording, sample_rate = pipeline.stop_recording()
        
        # Verify recording
        assert not pipeline.is_recording
        assert isinstance(recording, np.ndarray)
        assert recording.shape[0] == len(stereo_frame)
        assert recording.shape[1] == 2
        assert sample_rate == pipeline.sample_rate
    
    def test_stop_recording_not_recording(self, pipeline: AudioOutputPipeline) -> None:
        """Test stopping recording when not recording."""
        # Stop recording
        recording, sample_rate = pipeline.stop_recording()
        
        # Verify empty recording
        assert not pipeline.is_recording
        assert isinstance(recording, np.ndarray)
        assert recording.shape == (0, 2)
        assert sample_rate == pipeline.sample_rate
    
    @patch("soundfile.write")
    def test_save_audio_wav(self, mock_write: MagicMock, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test saving audio to WAV file."""
        # Mock os.makedirs
        with patch("os.makedirs") as mock_makedirs:
            # Save audio
            pipeline.save_audio(stereo_frame, 44100, "test.wav")
            
            # Verify makedirs was called
            mock_makedirs.assert_called_once()
            
            # Verify write was called
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert args[0] == "test.wav"
            assert np.array_equal(args[1], stereo_frame)
            assert args[2] == 44100
    
    def test_save_audio_unsupported_format(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test saving audio to unsupported format."""
        with pytest.raises(MusicGenerationError) as excinfo:
            pipeline.save_audio(stereo_frame, 44100, "test.xyz", "xyz")
        
        assert "Unsupported audio format" in str(excinfo.value)
    
    @patch("io.BytesIO")
    @patch("soundfile.write")
    def test_get_audio_as_bytes(
        self, mock_write: MagicMock, mock_bytesio: MagicMock, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray
    ) -> None:
        """Test getting audio as bytes."""
        # Mock BytesIO
        mock_buffer = MagicMock()
        mock_buffer.read.return_value = b"test_audio_data"
        mock_bytesio.return_value = mock_buffer
        
        # Get audio as bytes
        audio_bytes = pipeline.get_audio_as_bytes(stereo_frame, 44100)
        
        # Verify write was called
        mock_write.assert_called_once()
        
        # Verify bytes were returned
        assert audio_bytes == b"test_audio_data"
    
    def test_get_audio_as_bytes_unsupported_format(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test getting audio as bytes with unsupported format."""
        with pytest.raises(MusicGenerationError) as excinfo:
            pipeline.get_audio_as_bytes(stereo_frame, 44100, "xyz")
        
        assert "Unsupported audio format" in str(excinfo.value)
    
    def test_mix_audio_streams_empty(self, pipeline: AudioOutputPipeline) -> None:
        """Test mixing empty audio streams."""
        # Mix empty streams
        mixed, sample_rate = pipeline.mix_audio_streams([])
        
        # Verify empty result
        assert isinstance(mixed, np.ndarray)
        assert mixed.shape == (0, 2)
        assert sample_rate == pipeline.sample_rate
    
    def test_mix_audio_streams(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray, stereo_frame: np.ndarray) -> None:
        """Test mixing audio streams."""
        # Mix streams
        streams = [(mono_frame, 44100), (stereo_frame, 44100)]
        mixed, sample_rate = pipeline.mix_audio_streams(streams)
        
        # Verify mixed result
        assert isinstance(mixed, np.ndarray)
        assert mixed.shape[0] == max(len(mono_frame), len(stereo_frame))
        assert mixed.shape[1] == 2
        assert sample_rate == pipeline.sample_rate
    
    def test_mix_audio_streams_with_weights(
        self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray, stereo_frame: np.ndarray
    ) -> None:
        """Test mixing audio streams with weights."""
        # Mix streams with weights
        streams = [(mono_frame, 44100), (stereo_frame, 44100)]
        weights = [0.7, 0.3]
        mixed, sample_rate = pipeline.mix_audio_streams(streams, weights)
        
        # Verify mixed result
        assert isinstance(mixed, np.ndarray)
        assert mixed.shape[0] == max(len(mono_frame), len(stereo_frame))
        assert mixed.shape[1] == 2
        assert sample_rate == pipeline.sample_rate
    
    def test_apply_fade(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test applying fade to audio."""
        # Apply fade
        faded = pipeline.apply_fade(stereo_frame, fade_in=0.1, fade_out=0.1)
        
        # Verify faded audio
        assert isinstance(faded, np.ndarray)
        assert faded.shape == stereo_frame.shape
        
        # Check fade-in (first samples should be quieter)
        assert np.max(np.abs(faded[:100])) < np.max(np.abs(stereo_frame[:100]))
        
        # Check fade-out (last samples should be quieter)
        assert np.max(np.abs(faded[-100:])) < np.max(np.abs(stereo_frame[-100:]))
    
    def test_normalize_audio(self, pipeline: AudioOutputPipeline) -> None:
        """Test normalizing audio."""
        # Create test audio with varying amplitude
        audio = np.array([0.1, 0.5, -0.8, 0.3])
        
        # Normalize
        normalized = pipeline.normalize_audio(audio, target_level=0.9)
        
        # Verify normalization
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == audio.shape
        assert np.max(np.abs(normalized)) == pytest.approx(0.9)
    
    def test_normalize_audio_zero(self, pipeline: AudioOutputPipeline) -> None:
        """Test normalizing zero audio."""
        # Create zero audio
        audio = np.zeros(100)
        
        # Normalize
        normalized = pipeline.normalize_audio(audio)
        
        # Verify unchanged
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == audio.shape
        assert np.array_equal(normalized, audio)
    
    def test_resample_audio(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test resampling audio."""
        # Resample
        resampled = pipeline._resample_audio(mono_frame, 44100, 22050)
        
        # Verify resampled audio
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(mono_frame) // 2
    
    def test_resample_audio_stereo(self, pipeline: AudioOutputPipeline, stereo_frame: np.ndarray) -> None:
        """Test resampling stereo audio."""
        # Resample
        resampled = pipeline._resample_audio(stereo_frame, 44100, 22050)
        
        # Verify resampled audio
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(stereo_frame) // 2
        assert resampled.shape[1] == 2
    
    def test_resample_audio_same_rate(self, pipeline: AudioOutputPipeline, mono_frame: np.ndarray) -> None:
        """Test resampling audio to the same rate."""
        # Resample to same rate
        resampled = pipeline._resample_audio(mono_frame, 44100, 44100)
        
        # Verify unchanged
        assert isinstance(resampled, np.ndarray)
        assert np.array_equal(resampled, mono_frame)
    
    @patch("crowd_sentiment_music_generator.services.music_engine.audio_pipeline.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, pipeline: AudioOutputPipeline) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(pipeline.process_audio_frame, "__wrapped__")
        assert hasattr(pipeline.set_output_callback, "__wrapped__")
        assert hasattr(pipeline.get_current_buffer, "__wrapped__")
        assert hasattr(pipeline.start_recording, "__wrapped__")
        assert hasattr(pipeline.stop_recording, "__wrapped__")
        assert hasattr(pipeline.save_audio, "__wrapped__")
        assert hasattr(pipeline.get_audio_as_bytes, "__wrapped__")
        assert hasattr(pipeline.mix_audio_streams, "__wrapped__")
        assert hasattr(pipeline.apply_fade, "__wrapped__")
        assert hasattr(pipeline.normalize_audio, "__wrapped__")