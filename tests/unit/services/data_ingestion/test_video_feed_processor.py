"""Unit tests for video feed processor."""

import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import av
import m3u8
import requests

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.services.data_ingestion.video_feed_processor import (
    VideoFeedProcessor,
    AudioSegment,
)


class TestVideoFeedProcessor:
    """Test cases for VideoFeedProcessor class."""
    
    @pytest.fixture
    def processor(self) -> VideoFeedProcessor:
        """Create a VideoFeedProcessor instance for testing."""
        return VideoFeedProcessor(buffer_size=10)
    
    @pytest.fixture
    def sample_audio_segment(self) -> AudioSegment:
        """Create a sample audio segment."""
        return AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=100.0,
            duration=1.0
        )
    
    @pytest.fixture
    def sample_hls_playlist(self) -> str:
        """Create a sample HLS playlist."""
        return """
        #EXTM3U
        #EXT-X-VERSION:3
        #EXT-X-TARGETDURATION:10
        #EXT-X-MEDIA-SEQUENCE:0
        
        #EXTINF:9.0,
        segment1.ts
        #EXTINF:9.0,
        segment2.ts
        #EXTINF:9.0,
        segment3.ts
        
        #EXT-X-ENDLIST
        """
    
    def test_initialization(self, processor: VideoFeedProcessor) -> None:
        """Test processor initialization."""
        assert processor.buffer_size == 10
        assert processor.audio_buffer == []
    
    @patch("requests.get")
    def test_process_hls_playlist(
        self, mock_get: MagicMock, processor: VideoFeedProcessor, sample_hls_playlist: str
    ) -> None:
        """Test processing HLS playlist."""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = sample_hls_playlist
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Process playlist
        segments = processor.process_hls_playlist("https://example.com/playlist.m3u8")
        
        # Verify segments
        assert len(segments) == 3
        assert segments[0]["url"] == "https://example.com/segment1.ts"
        assert segments[0]["timestamp"] == 0.0
        assert segments[0]["duration"] == 9.0
        assert segments[1]["url"] == "https://example.com/segment2.ts"
        assert segments[1]["timestamp"] == 9.0
        assert segments[1]["duration"] == 9.0
        assert segments[2]["url"] == "https://example.com/segment3.ts"
        assert segments[2]["timestamp"] == 18.0
        assert segments[2]["duration"] == 9.0
    
    @patch("requests.get")
    def test_process_hls_playlist_absolute_urls(
        self, mock_get: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing HLS playlist with absolute URLs."""
        # Create playlist with absolute URLs
        playlist = """
        #EXTM3U
        #EXT-X-VERSION:3
        #EXT-X-TARGETDURATION:10
        #EXT-X-MEDIA-SEQUENCE:0
        
        #EXTINF:9.0,
        https://cdn.example.com/segment1.ts
        #EXTINF:9.0,
        https://cdn.example.com/segment2.ts
        
        #EXT-X-ENDLIST
        """
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = playlist
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Process playlist
        segments = processor.process_hls_playlist("https://example.com/playlist.m3u8")
        
        # Verify segments
        assert len(segments) == 2
        assert segments[0]["url"] == "https://cdn.example.com/segment1.ts"
        assert segments[1]["url"] == "https://cdn.example.com/segment2.ts"
    
    @patch("requests.get")
    def test_process_hls_playlist_request_error(
        self, mock_get: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing HLS playlist with request error."""
        # Mock response that raises an exception
        mock_get.side_effect = requests.RequestException("Request failed")
        
        # Process playlist should raise an exception
        with pytest.raises(AudioProcessingError) as excinfo:
            processor.process_hls_playlist("https://example.com/playlist.m3u8")
        
        # Verify error message
        assert "Failed to fetch HLS playlist" in str(excinfo.value)
    
    @patch("requests.get")
    def test_process_hls_playlist_parse_error(
        self, mock_get: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing HLS playlist with parse error."""
        # Mock response with invalid playlist
        mock_response = MagicMock()
        mock_response.text = "Invalid playlist"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Process playlist should raise an exception
        with pytest.raises(AudioProcessingError) as excinfo:
            processor.process_hls_playlist("https://example.com/playlist.m3u8")
        
        # Verify error message
        assert "Failed to parse HLS playlist" in str(excinfo.value)
    
    @patch("av.open")
    def test_process_chunk(self, mock_av_open: MagicMock, processor: VideoFeedProcessor) -> None:
        """Test processing video chunk."""
        # Create mock audio frame
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = np.zeros((1000, 2))
        
        # Create mock audio stream
        mock_stream = MagicMock()
        mock_stream.type = "audio"
        mock_stream.rate = 44100
        
        # Create mock container
        mock_container = MagicMock()
        mock_container.streams = [mock_stream]
        mock_container.decode.return_value = [mock_frame]
        
        # Configure mock_av_open to return our mock container
        mock_av_open.return_value = mock_container
        
        # Process chunk
        audio_data, sample_rate = processor.process_chunk(b"video_data", 100.0)
        
        # Verify audio data
        assert isinstance(audio_data, np.ndarray)
        assert sample_rate == 44100
        
        # Verify buffer
        assert len(processor.audio_buffer) == 1
        assert processor.audio_buffer[0].timestamp == 100.0
    
    @patch("av.open")
    def test_process_chunk_no_audio_stream(
        self, mock_av_open: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing video chunk with no audio stream."""
        # Create mock container with no audio stream
        mock_container = MagicMock()
        mock_container.streams = []
        
        # Configure mock_av_open to return our mock container
        mock_av_open.return_value = mock_container
        
        # Process chunk should raise an exception
        with pytest.raises(AudioProcessingError) as excinfo:
            processor.process_chunk(b"video_data", 100.0)
        
        # Verify error message
        assert "No audio stream found" in str(excinfo.value)
    
    @patch("av.open")
    def test_process_chunk_no_audio_frames(
        self, mock_av_open: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing video chunk with no audio frames."""
        # Create mock audio stream
        mock_stream = MagicMock()
        mock_stream.type = "audio"
        
        # Create mock container with no frames
        mock_container = MagicMock()
        mock_container.streams = [mock_stream]
        mock_container.decode.return_value = []
        
        # Configure mock_av_open to return our mock container
        mock_av_open.return_value = mock_container
        
        # Process chunk should raise an exception
        with pytest.raises(AudioProcessingError) as excinfo:
            processor.process_chunk(b"video_data", 100.0)
        
        # Verify error message
        assert "No audio frames found" in str(excinfo.value)
    
    @patch("av.open")
    def test_process_chunk_av_error(
        self, mock_av_open: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test processing video chunk with AV error."""
        # Configure mock_av_open to raise an exception
        mock_av_open.side_effect = av.AVError("Decoding failed")
        
        # Process chunk should raise an exception
        with pytest.raises(AudioProcessingError) as excinfo:
            processor.process_chunk(b"video_data", 100.0)
        
        # Verify error message
        assert "Failed to decode video chunk" in str(excinfo.value)
    
    @patch("scipy.signal.butter")
    @patch("scipy.signal.filtfilt")
    def test_extract_crowd_audio(
        self, mock_filtfilt: MagicMock, mock_butter: MagicMock, processor: VideoFeedProcessor
    ) -> None:
        """Test extracting crowd audio."""
        # Mock butter and filtfilt
        mock_butter.return_value = (1, 2)
        mock_filtfilt.return_value = np.ones(1000)
        
        # Extract crowd audio from mono
        mono_audio = np.zeros(1000)
        crowd_audio = processor.extract_crowd_audio(mono_audio)
        
        # Verify crowd audio
        assert isinstance(crowd_audio, np.ndarray)
        assert len(crowd_audio) == 1000
        assert np.array_equal(crowd_audio, np.ones(1000))
        
        # Extract crowd audio from stereo
        stereo_audio = np.zeros((1000, 2))
        crowd_audio = processor.extract_crowd_audio(stereo_audio)
        
        # Verify crowd audio
        assert isinstance(crowd_audio, np.ndarray)
        assert len(crowd_audio) == 1000
        assert np.array_equal(crowd_audio, np.ones(1000))
    
    def test_extract_crowd_audio_error(self, processor: VideoFeedProcessor) -> None:
        """Test extracting crowd audio with error."""
        # Mock scipy.signal to raise an exception
        with patch("scipy.signal.butter", side_effect=Exception("Processing failed")):
            # Extract crowd audio should raise an exception
            with pytest.raises(AudioProcessingError) as excinfo:
                processor.extract_crowd_audio(np.zeros(1000))
            
            # Verify error message
            assert "Failed to isolate crowd noise" in str(excinfo.value)
    
    def test_get_audio_segment_at_timestamp(
        self, processor: VideoFeedProcessor, sample_audio_segment: AudioSegment
    ) -> None:
        """Test getting audio segment at timestamp."""
        # Add segment to buffer
        processor.audio_buffer = [sample_audio_segment]
        
        # Get segment at timestamp
        segment = processor.get_audio_segment_at_timestamp(100.5)
        
        # Verify segment
        assert segment == sample_audio_segment
        
        # Get segment at timestamp outside range
        segment = processor.get_audio_segment_at_timestamp(102.0)
        
        # Verify no segment found
        assert segment is None
    
    def test_get_audio_segments_in_range(
        self, processor: VideoFeedProcessor, sample_audio_segment: AudioSegment
    ) -> None:
        """Test getting audio segments in range."""
        # Create multiple segments
        segment1 = sample_audio_segment
        segment2 = AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=102.0,
            duration=1.0
        )
        segment3 = AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=104.0,
            duration=1.0
        )
        
        # Add segments to buffer
        processor.audio_buffer = [segment1, segment2, segment3]
        
        # Get segments in range
        segments = processor.get_audio_segments_in_range(101.5, 103.5)
        
        # Verify segments
        assert len(segments) == 1
        assert segments[0] == segment2
        
        # Get segments in wider range
        segments = processor.get_audio_segments_in_range(99.5, 104.5)
        
        # Verify segments
        assert len(segments) == 3
        assert segments[0] == segment1
        assert segments[1] == segment2
        assert segments[2] == segment3
    
    def test_clear_buffer(self, processor: VideoFeedProcessor, sample_audio_segment: AudioSegment) -> None:
        """Test clearing buffer."""
        # Add segment to buffer
        processor.audio_buffer = [sample_audio_segment]
        
        # Clear buffer
        processor.clear_buffer()
        
        # Verify buffer is empty
        assert processor.audio_buffer == []
    
    def test_add_to_buffer(self, processor: VideoFeedProcessor) -> None:
        """Test adding to buffer."""
        # Create segments
        segment1 = AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=100.0,
            duration=1.0
        )
        segment2 = AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=102.0,
            duration=1.0
        )
        segment3 = AudioSegment(
            audio_data=np.zeros(1000),
            sample_rate=44100,
            timestamp=95.0,  # Old segment
            duration=1.0
        )
        
        # Add segments to buffer
        processor._add_to_buffer(segment1)
        processor._add_to_buffer(segment2)
        
        # Verify buffer
        assert len(processor.audio_buffer) == 2
        assert processor.audio_buffer[0] == segment1
        assert processor.audio_buffer[1] == segment2
        
        # Add old segment (should be removed due to buffer size)
        processor._add_to_buffer(segment3)
        
        # Verify buffer (segment3 should be removed since it's older than cutoff time)
        assert len(processor.audio_buffer) == 2
        assert processor.audio_buffer[0] == segment1
        assert processor.audio_buffer[1] == segment2
    
    @patch("crowd_sentiment_music_generator.services.data_ingestion.video_feed_processor.with_error_handling")
    def test_error_handling(self, mock_error_handler: MagicMock, processor: VideoFeedProcessor) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Configure the mock to pass through the original function
        mock_error_handler.side_effect = lambda f: f
        
        # Verify error handling is applied to public methods
        assert hasattr(processor.process_hls_playlist, "__wrapped__")
        assert hasattr(processor.process_chunk, "__wrapped__")
        assert hasattr(processor.extract_crowd_audio, "__wrapped__")
        assert hasattr(processor.get_audio_segment_at_timestamp, "__wrapped__")
        assert hasattr(processor.get_audio_segments_in_range, "__wrapped__")
        assert hasattr(processor.clear_buffer, "__wrapped__")