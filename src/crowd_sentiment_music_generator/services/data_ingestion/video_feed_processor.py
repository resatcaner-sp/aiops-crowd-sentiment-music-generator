"""Video feed processor for HLS streams."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import av
import m3u8
import requests
from pydantic import BaseModel

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class AudioSegment(BaseModel):
    """Model for an audio segment extracted from video.
    
    Attributes:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        timestamp: HLS timestamp of the segment
        duration: Duration of the segment in seconds
    """
    
    audio_data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float


class VideoFeedProcessor:
    """Processes incoming video feed to extract audio and timestamps.
    
    This class handles HLS stream processing, audio extraction, and buffer
    management for delayed processing.
    
    Attributes:
        buffer_size: Size of the audio buffer in seconds
        audio_buffer: Buffer of recent audio segments
        logger: Logger instance
    """
    
    def __init__(self, buffer_size: int = 30) -> None:
        """Initialize the video feed processor.
        
        Args:
            buffer_size: Size of the audio buffer in seconds
        """
        self.buffer_size = buffer_size
        self.audio_buffer: List[AudioSegment] = []
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    def process_hls_playlist(self, playlist_url: str) -> List[Dict[str, Union[str, float]]]:
        """Process an HLS playlist to get segment information.
        
        Args:
            playlist_url: URL of the HLS playlist
            
        Returns:
            List of dictionaries containing segment URLs and timestamps
            
        Raises:
            AudioProcessingError: If playlist processing fails
        """
        try:
            # Get playlist content
            response = requests.get(playlist_url)
            response.raise_for_status()
            
            # Parse playlist
            playlist = m3u8.loads(response.text)
            
            # Extract segment information
            segments = []
            current_timestamp = 0.0
            
            for segment in playlist.segments:
                # Calculate absolute URL if segment URL is relative
                if not segment.uri.startswith(("http://", "https://")):
                    base_url = playlist_url.rsplit("/", 1)[0]
                    segment_url = f"{base_url}/{segment.uri}"
                else:
                    segment_url = segment.uri
                
                # Add segment info
                segments.append({
                    "url": segment_url,
                    "timestamp": current_timestamp,
                    "duration": segment.duration
                })
                
                # Update timestamp for next segment
                current_timestamp += segment.duration
            
            return segments
        except requests.RequestException as e:
            raise AudioProcessingError(f"Failed to fetch HLS playlist: {str(e)}")
        except m3u8.ParseError as e:
            raise AudioProcessingError(f"Failed to parse HLS playlist: {str(e)}")
        except Exception as e:
            raise AudioProcessingError(f"Error processing HLS playlist: {str(e)}")
    
    @with_error_handling
    def process_chunk(self, video_chunk: bytes, hls_timestamp: float) -> Tuple[np.ndarray, float]:
        """Process a video chunk and extract audio and timestamp.
        
        Args:
            video_chunk: Raw video chunk data
            hls_timestamp: HLS timestamp of the chunk
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If audio extraction fails
        """
        try:
            # Create in-memory container
            container = av.open(video_chunk)
            
            # Get audio stream
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if not audio_stream:
                raise AudioProcessingError("No audio stream found in video chunk")
            
            # Get sample rate
            sample_rate = audio_stream.rate
            
            # Extract audio frames
            audio_frames = []
            for frame in container.decode(audio_stream):
                audio_frames.append(frame.to_ndarray())
            
            if not audio_frames:
                raise AudioProcessingError("No audio frames found in video chunk")
            
            # Concatenate frames
            audio_data = np.concatenate(audio_frames)
            
            # Get duration
            duration = float(len(audio_data)) / sample_rate
            
            # Create audio segment
            segment = AudioSegment(
                audio_data=audio_data,
                sample_rate=sample_rate,
                timestamp=hls_timestamp,
                duration=duration
            )
            
            # Add to buffer
            self._add_to_buffer(segment)
            
            return audio_data, sample_rate
        except av.AVError as e:
            raise AudioProcessingError(f"Failed to decode video chunk: {str(e)}")
        except Exception as e:
            raise AudioProcessingError(f"Error processing video chunk: {str(e)}")
    
    @with_error_handling
    def extract_crowd_audio(self, audio: np.ndarray) -> np.ndarray:
        """Isolate crowd noise from commentary and other sounds.
        
        This method uses a combination of frequency filtering and
        source separation to isolate crowd noise.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Isolated crowd noise as numpy array
            
        Raises:
            AudioProcessingError: If crowd noise isolation fails
        """
        try:
            # Simple bandpass filter for crowd noise (typically 100-3000 Hz)
            # This is a simplified approach - a real implementation would use
            # more sophisticated source separation techniques
            from scipy import signal
            
            # Get sample rate (assuming mono audio)
            if len(audio.shape) > 1:
                # Convert stereo to mono by averaging channels
                audio = np.mean(audio, axis=1)
            
            # Apply bandpass filter
            # Assuming sample rate of 44100 Hz
            sample_rate = 44100
            nyquist = 0.5 * sample_rate
            low = 100 / nyquist
            high = 3000 / nyquist
            b, a = signal.butter(4, [low, high], btype="band")
            crowd_audio = signal.filtfilt(b, a, audio)
            
            return crowd_audio
        except Exception as e:
            raise AudioProcessingError(f"Failed to isolate crowd noise: {str(e)}")
    
    @with_error_handling
    def get_audio_segment_at_timestamp(self, timestamp: float) -> Optional[AudioSegment]:
        """Get the audio segment at a specific timestamp.
        
        Args:
            timestamp: Timestamp to look for
            
        Returns:
            AudioSegment if found, None otherwise
        """
        for segment in self.audio_buffer:
            segment_start = segment.timestamp
            segment_end = segment.timestamp + segment.duration
            
            if segment_start <= timestamp < segment_end:
                return segment
        
        return None
    
    @with_error_handling
    def get_audio_segments_in_range(
        self, start_time: float, end_time: float
    ) -> List[AudioSegment]:
        """Get all audio segments in a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of AudioSegment objects in the range
        """
        segments = []
        
        for segment in self.audio_buffer:
            segment_start = segment.timestamp
            segment_end = segment.timestamp + segment.duration
            
            # Check if segment overlaps with the requested range
            if segment_end > start_time and segment_start < end_time:
                segments.append(segment)
        
        return segments
    
    @with_error_handling
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self.audio_buffer = []
    
    def _add_to_buffer(self, segment: AudioSegment) -> None:
        """Add an audio segment to the buffer and maintain buffer size.
        
        Args:
            segment: AudioSegment to add
        """
        # Add segment to buffer
        self.audio_buffer.append(segment)
        
        # Sort buffer by timestamp
        self.audio_buffer.sort(key=lambda s: s.timestamp)
        
        # Remove old segments to maintain buffer size
        current_time = segment.timestamp
        cutoff_time = current_time - self.buffer_size
        
        # Remove segments older than cutoff time
        self.audio_buffer = [s for s in self.audio_buffer if s.timestamp >= cutoff_time]