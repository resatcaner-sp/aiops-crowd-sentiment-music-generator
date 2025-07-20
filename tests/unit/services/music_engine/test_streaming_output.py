"""Unit tests for streaming output module."""

import pytest
import os
from unittest.mock import MagicMock, patch
import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.music_engine.audio_pipeline import AudioOutputPipeline
from crowd_sentiment_music_generator.services.music_engine.streaming_output import StreamingAudioOutput, AudioFormatConverter


class TestStreamingAudioOutput:
    """Test cases for StreamingAudioOutput class."""
    
    @pytest.fixture
    def mock_pipeline(self) -> MagicMock:
        """Create a mock audio pipeline."""
        pipeline = MagicMock(spec=AudioOutputPipeline)
        pipeline.sample_rate = 44100
        pipeline.channels = 2
        pipeline.buffer_position = 0
        pipeline.get_current_buffer.return_value = (np.zeros((1000, 2)), 44100)
        pipeline.get_audio_as_bytes.return_value = b"test_audio_data"
        pipeline.SUPPORTED_FORMATS = {
            "wav": {"extension": ".wav"},
            "mp3": {"extension": ".mp3"}
        }
        return pipeline
    
    @pytest.fixture
    def streaming(self, mock_pipeline: MagicMock) -> StreamingAudioOutput:
        """Create a StreamingAudioOutput instance for testing."""
        config = SystemConfig()
        return StreamingAudioOutput(pipeline=mock_pipeline, config=config)
    
    def test_initialization(self, streaming: StreamingAudioOutput, mock_pipeline: MagicMock) -> None:
        """Test streaming initialization."""
        assert streaming is not None
        assert streaming.config is not None
        assert streaming.pipeline == mock_pipeline
        assert not streaming.is_streaming
        assert streaming.streaming_thread is None
        assert streaming.stream_clients == []
        assert streaming.stream_format == "wav"
    
    def test_start_streaming(self, streaming: StreamingAudioOutput) -> None:
        """Test starting streaming."""
        # Start streaming
        streaming.start_streaming()
        
        # Verify streaming state
        assert streaming.is_streaming
        assert streaming.streaming_thread is not None
        assert streaming.streaming_thread.is_alive()
        
        # Clean up
        streaming.stop_streaming()
    
    def test_start_streaming_already_streaming(self, streaming: StreamingAudioOutput) -> None:
        """Test starting streaming when already streaming."""
        # Start streaming
        streaming.start_streaming()
        
        # Get thread
        thread = streaming.streaming_thread
        
        # Start streaming again
        streaming.start_streaming()
        
        # Verify streaming state
        assert streaming.is_streaming
        assert streaming.streaming_thread == thread
        
        # Clean up
        streaming.stop_streaming()
    
    def test_start_streaming_unsupported_format(self, streaming: StreamingAudioOutput) -> None:
        """Test starting streaming with unsupported format."""
        with pytest.raises(MusicGenerationError) as excinfo:
            streaming.start_streaming("xyz")
        
        assert "Unsupported audio format" in str(excinfo.value)
        assert not streaming.is_streaming
    
    def test_stop_streaming(self, streaming: StreamingAudioOutput) -> None:
        """Test stopping streaming."""
        # Start streaming
        streaming.start_streaming()
        
        # Stop streaming
        streaming.stop_streaming()
        
        # Verify streaming state
        assert not streaming.is_streaming
        assert not streaming.streaming_thread.is_alive()
    
    def test_stop_streaming_not_streaming(self, streaming: StreamingAudioOutput) -> None:
        """Test stopping streaming when not streaming."""
        # Stop streaming
        streaming.stop_streaming()
        
        # Verify streaming state
        assert not streaming.is_streaming
    
    def test_add_stream_client(self, streaming: StreamingAudioOutput) -> None:
        """Test adding a stream client."""
        # Create mock callback
        callback = MagicMock()
        
        # Add client
        streaming.add_stream_client("test_client", callback)
        
        # Verify client was added
        assert len(streaming.stream_clients) == 1
        assert streaming.stream_clients[0][0] == "test_client"
        assert streaming.stream_clients[0][1] == callback
        
        # Verify streaming was started
        assert streaming.is_streaming
        
        # Clean up
        streaming.stop_streaming()
    
    def test_add_stream_client_existing(self, streaming: StreamingAudioOutput) -> None:
        """Test adding an existing stream client."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Add client
        streaming.add_stream_client("test_client", callback1)
        
        # Add same client with different callback
        streaming.add_stream_client("test_client", callback2)
        
        # Verify client was updated
        assert len(streaming.stream_clients) == 1
        assert streaming.stream_clients[0][0] == "test_client"
        assert streaming.stream_clients[0][1] == callback2
        
        # Clean up
        streaming.stop_streaming()
    
    def test_remove_stream_client(self, streaming: StreamingAudioOutput) -> None:
        """Test removing a stream client."""
        # Create mock callback
        callback = MagicMock()
        
        # Add client
        streaming.add_stream_client("test_client", callback)
        
        # Remove client
        streaming.remove_stream_client("test_client")
        
        # Verify client was removed
        assert len(streaming.stream_clients) == 0
        
        # Verify streaming was stopped
        assert not streaming.is_streaming
    
    def test_remove_stream_client_nonexistent(self, streaming: StreamingAudioOutput) -> None:
        """Test removing a nonexistent stream client."""
        # Remove nonexistent client
        streaming.remove_stream_client("nonexistent_client")
        
        # Verify no change
        assert len(streaming.stream_clients) == 0
    
    def test_get_stream_info(self, streaming: StreamingAudioOutput) -> None:
        """Test getting stream information."""
        # Get info
        info = streaming.get_stream_info()
        
        # Verify info
        assert isinstance(info, dict)
        assert "is_streaming" in info
        assert "format" in info
        assert "sample_rate" in info
        assert "channels" in info
        assert "chunk_size" in info
        assert "client_count" in info
    
    @patch("time.sleep")
    def test_streaming_loop(self, mock_sleep: MagicMock, streaming: StreamingAudioOutput, mock_pipeline: MagicMock) -> None:
        """Test the streaming loop."""
        # Set up mock pipeline to simulate buffer position change
        mock_pipeline.buffer_position = 500
        
        # Set up streaming
        streaming.is_streaming = True
        
        # Run one iteration of the loop
        streaming._streaming_loop()
        
        # Verify pipeline methods were called
        mock_pipeline.get_current_buffer.assert_called_once()
        mock_pipeline.get_audio_as_bytes.assert_called_once()
    
    def test_send_chunk_to_clients(self, streaming: StreamingAudioOutput, mock_pipeline: MagicMock) -> None:
        """Test sending chunks to clients."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Add clients
        streaming.stream_clients = [("client1", callback1), ("client2", callback2)]
        
        # Send chunk
        chunk = np.zeros((1000, 2))
        streaming._send_chunk_to_clients(chunk, 44100)
        
        # Verify pipeline method was called
        mock_pipeline.get_audio_as_bytes.assert_called_once()
        
        # Verify callbacks were called
        callback1.assert_called_once_with(b"test_audio_data")
        callback2.assert_called_once_with(b"test_audio_data")
    
    def test_send_chunk_to_clients_error(self, streaming: StreamingAudioOutput, mock_pipeline: MagicMock) -> None:
        """Test sending chunks to clients with error."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock(side_effect=Exception("Test error"))
        
        # Add clients
        streaming.stream_clients = [("client1", callback1), ("client2", callback2)]
        
        # Send chunk
        chunk = np.zeros((1000, 2))
        streaming._send_chunk_to_clients(chunk, 44100)
        
        # Verify pipeline method was called
        mock_pipeline.get_audio_as_bytes.assert_called_once()
        
        # Verify first callback was called
        callback1.assert_called_once_with(b"test_audio_data")
        
        # Second callback should have been called but raised an exception
        callback2.assert_called_once_with(b"test_audio_data")
    
    @patch("crowd_sentiment_music_generator.services.music_engine.streaming_output.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, streaming: StreamingAudioOutput) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(streaming.start_streaming, "__wrapped__")
        assert hasattr(streaming.stop_streaming, "__wrapped__")
        assert hasattr(streaming.add_stream_client, "__wrapped__")
        assert hasattr(streaming.remove_stream_client, "__wrapped__")
        assert hasattr(streaming.get_stream_info, "__wrapped__")


class TestAudioFormatConverter:
    """Test cases for AudioFormatConverter class."""
    
    @pytest.fixture
    def converter(self) -> AudioFormatConverter:
        """Create an AudioFormatConverter instance for testing."""
        return AudioFormatConverter()
    
    @pytest.fixture
    def mono_audio(self) -> np.ndarray:
        """Create mono audio for testing."""
        # 0.5 seconds of 440 Hz sine wave at 44100 Hz
        t = np.linspace(0, 0.5, int(44100 * 0.5), False)
        return np.sin(2 * np.pi * 440 * t)
    
    @pytest.fixture
    def stereo_audio(self) -> np.ndarray:
        """Create stereo audio for testing."""
        # 0.5 seconds of 440 Hz sine wave at 44100 Hz
        t = np.linspace(0, 0.5, int(44100 * 0.5), False)
        mono = np.sin(2 * np.pi * 440 * t)
        return np.column_stack((mono, mono))
    
    def test_initialization(self, converter: AudioFormatConverter) -> None:
        """Test converter initialization."""
        assert converter is not None
    
    @patch("soundfile.read")
    @patch("soundfile.write")
    def test_convert_format(
        self, mock_write: MagicMock, mock_read: MagicMock, converter: AudioFormatConverter, stereo_audio: np.ndarray
    ) -> None:
        """Test converting audio format."""
        # Mock soundfile.read
        mock_read.return_value = (stereo_audio, 44100)
        
        # Convert format
        converter.convert_format("input.wav", "output.flac", "flac")
        
        # Verify read was called
        mock_read.assert_called_once_with("input.wav")
        
        # Verify write was called
        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        assert args[0] == "output.flac"
        assert np.array_equal(args[1], stereo_audio)
        assert args[2] == 44100
    
    @patch("soundfile.read")
    def test_convert_format_unsupported(
        self, mock_read: MagicMock, converter: AudioFormatConverter, stereo_audio: np.ndarray
    ) -> None:
        """Test converting to unsupported format."""
        # Mock soundfile.read
        mock_read.return_value = (stereo_audio, 44100)
        
        with pytest.raises(MusicGenerationError) as excinfo:
            converter.convert_format("input.wav", "output.xyz", "xyz")
        
        assert "Unsupported output format" in str(excinfo.value)
    
    def test_convert_sample_rate(self, converter: AudioFormatConverter, mono_audio: np.ndarray) -> None:
        """Test converting sample rate."""
        # Convert sample rate
        resampled = converter.convert_sample_rate(mono_audio, 44100, 22050)
        
        # Verify resampled audio
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(mono_audio) // 2
    
    def test_convert_channels_mono_to_stereo(self, converter: AudioFormatConverter, mono_audio: np.ndarray) -> None:
        """Test converting mono to stereo."""
        # Convert channels
        stereo = converter.convert_channels(mono_audio, 2)
        
        # Verify stereo audio
        assert isinstance(stereo, np.ndarray)
        assert stereo.shape[1] == 2
        assert np.array_equal(stereo[:, 0], stereo[:, 1])
    
    def test_convert_channels_stereo_to_mono(self, converter: AudioFormatConverter, stereo_audio: np.ndarray) -> None:
        """Test converting stereo to mono."""
        # Convert channels
        mono = converter.convert_channels(stereo_audio, 1)
        
        # Verify mono audio
        assert isinstance(mono, np.ndarray)
        assert len(mono.shape) == 1 or mono.shape[1] == 1
    
    def test_resample_audio(self, converter: AudioFormatConverter, mono_audio: np.ndarray) -> None:
        """Test resampling audio."""
        # Resample
        resampled = converter._resample_audio(mono_audio, 44100, 22050)
        
        # Verify resampled audio
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(mono_audio) // 2
    
    def test_convert_channels(self, converter: AudioFormatConverter, mono_audio: np.ndarray) -> None:
        """Test converting channels."""
        # Convert to stereo
        stereo = converter._convert_channels(mono_audio, 2)
        
        # Verify stereo audio
        assert isinstance(stereo, np.ndarray)
        assert stereo.shape[1] == 2
        
        # Convert back to mono
        mono = converter._convert_channels(stereo, 1)
        
        # Verify mono audio
        assert isinstance(mono, np.ndarray)
        assert len(mono.shape) == 1 or mono.shape[1] == 1
    
    @patch("crowd_sentiment_music_generator.services.music_engine.streaming_output.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, converter: AudioFormatConverter) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(converter.convert_format, "__wrapped__")
        assert hasattr(converter.convert_sample_rate, "__wrapped__")
        assert hasattr(converter.convert_channels, "__wrapped__")