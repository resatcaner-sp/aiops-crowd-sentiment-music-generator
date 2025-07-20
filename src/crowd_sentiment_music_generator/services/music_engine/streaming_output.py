"""Streaming audio output module.

This module provides functionality for streaming audio output to various destinations,
such as web clients, audio devices, or files.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Callable

import numpy as np

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.music_engine.audio_pipeline import AudioOutputPipeline
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class StreamingAudioOutput:
    """Manages streaming audio output for real-time playback.
    
    This class provides methods for streaming audio output to various destinations,
    such as web clients, audio devices, or files.
    """
    
    def __init__(
        self, 
        pipeline: Optional[AudioOutputPipeline] = None,
        config: Optional[SystemConfig] = None
    ):
        """Initialize the streaming audio output.
        
        Args:
            pipeline: Audio output pipeline (optional, creates a new one if not provided)
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.pipeline = pipeline or AudioOutputPipeline(self.config)
        self.is_streaming = False
        self.streaming_thread = None
        self.stream_clients = []
        self.stream_lock = threading.Lock()
        self.chunk_size = 1024  # Samples per chunk
        self.stream_format = "wav"  # Default stream format
        logger.info("Initialized StreamingAudioOutput")
    
    @with_error_handling
    def start_streaming(self, format: str = "wav") -> None:
        """Start streaming audio output.
        
        Args:
            format: Audio format for streaming (wav, flac, ogg, mp3)
            
        Raises:
            MusicGenerationError: If streaming start fails
        """
        if self.is_streaming:
            logger.warning("Streaming already in progress")
            return
        
        try:
            # Check if format is supported
            if format not in self.pipeline.SUPPORTED_FORMATS:
                raise MusicGenerationError(f"Unsupported audio format: {format}")
            
            # Set stream format
            self.stream_format = format
            
            # Start streaming
            self.is_streaming = True
            self.streaming_thread = threading.Thread(
                target=self._streaming_loop,
                daemon=True
            )
            self.streaming_thread.start()
            
            logger.info(f"Started streaming in {format} format")
        
        except Exception as e:
            self.is_streaming = False
            raise MusicGenerationError(f"Failed to start streaming: {str(e)}")
    
    @with_error_handling
    def stop_streaming(self) -> None:
        """Stop streaming audio output.
        
        Raises:
            MusicGenerationError: If streaming stop fails
        """
        if not self.is_streaming:
            logger.warning("No streaming in progress")
            return
        
        try:
            # Stop streaming
            self.is_streaming = False
            
            # Wait for thread to finish (with timeout)
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=2.0)
            
            # Clear clients
            with self.stream_lock:
                self.stream_clients = []
            
            logger.info("Stopped streaming")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to stop streaming: {str(e)}")
    
    @with_error_handling
    def add_stream_client(self, client_id: str, callback: Callable[[bytes], None]) -> None:
        """Add a client to receive streaming audio.
        
        Args:
            client_id: Unique client identifier
            callback: Function that accepts audio data as bytes
            
        Raises:
            MusicGenerationError: If client addition fails
        """
        try:
            # Add client to list
            with self.stream_lock:
                # Check if client already exists
                for i, (cid, _) in enumerate(self.stream_clients):
                    if cid == client_id:
                        # Replace existing client
                        self.stream_clients[i] = (client_id, callback)
                        logger.debug(f"Updated stream client: {client_id}")
                        return
                
                # Add new client
                self.stream_clients.append((client_id, callback))
            
            logger.debug(f"Added stream client: {client_id}")
            
            # Start streaming if not already started
            if not self.is_streaming:
                self.start_streaming(self.stream_format)
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to add stream client: {str(e)}")
    
    @with_error_handling
    def remove_stream_client(self, client_id: str) -> None:
        """Remove a client from streaming audio.
        
        Args:
            client_id: Unique client identifier
            
        Raises:
            MusicGenerationError: If client removal fails
        """
        try:
            # Remove client from list
            with self.stream_lock:
                self.stream_clients = [(cid, cb) for cid, cb in self.stream_clients if cid != client_id]
            
            logger.debug(f"Removed stream client: {client_id}")
            
            # Stop streaming if no clients left
            if not self.stream_clients and self.is_streaming:
                self.stop_streaming()
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to remove stream client: {str(e)}")
    
    @with_error_handling
    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the current stream.
        
        Returns:
            Dictionary with stream information
            
        Raises:
            MusicGenerationError: If information retrieval fails
        """
        try:
            info = {
                "is_streaming": self.is_streaming,
                "format": self.stream_format,
                "sample_rate": self.pipeline.sample_rate,
                "channels": self.pipeline.channels,
                "chunk_size": self.chunk_size,
                "client_count": len(self.stream_clients)
            }
            
            return info
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to get stream info: {str(e)}")
    
    def _streaming_loop(self) -> None:
        """Main streaming loop that runs in a separate thread."""
        logger.info("Streaming loop started")
        
        last_position = 0
        
        while self.is_streaming:
            try:
                # Get current buffer
                buffer, sample_rate = self.pipeline.get_current_buffer()
                
                # Find new audio since last position
                current_position = self.pipeline.buffer_position
                
                if current_position != last_position:
                    # Calculate chunk to send
                    if current_position > last_position:
                        # No wrap-around
                        chunk = buffer[last_position:current_position]
                    else:
                        # Wrap-around
                        chunk = np.vstack((
                            buffer[last_position:],
                            buffer[:current_position]
                        ))
                    
                    # Update last position
                    last_position = current_position
                    
                    # Send chunk to clients
                    if len(chunk) > 0:
                        self._send_chunk_to_clients(chunk, sample_rate)
            
            except Exception as e:
                logger.error(f"Error in streaming loop: {str(e)}")
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.01)
        
        logger.info("Streaming loop stopped")
    
    def _send_chunk_to_clients(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Send an audio chunk to all clients.
        
        Args:
            chunk: Audio chunk as numpy array
            sample_rate: Sample rate of the audio
        """
        try:
            # Convert chunk to bytes in the specified format
            chunk_bytes = self.pipeline.get_audio_as_bytes(chunk, sample_rate, self.stream_format)
            
            # Send to all clients
            with self.stream_lock:
                for client_id, callback in self.stream_clients:
                    try:
                        callback(chunk_bytes)
                    except Exception as e:
                        logger.error(f"Error sending to client {client_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error sending chunk to clients: {str(e)}")


class AudioFormatConverter:
    """Provides utilities for audio format conversion.
    
    This class provides methods for converting audio between different formats
    and sample rates.
    """
    
    def __init__(self):
        """Initialize the audio format converter."""
        logger.info("Initialized AudioFormatConverter")
    
    @with_error_handling
    def convert_format(
        self, 
        input_file: str, 
        output_file: str, 
        output_format: str = None,
        sample_rate: int = None,
        channels: int = None,
        bit_depth: int = None
    ) -> None:
        """Convert an audio file to a different format.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            output_format: Output format (wav, flac, ogg, mp3)
            sample_rate: Output sample rate (optional)
            channels: Output channel count (optional)
            bit_depth: Output bit depth (optional)
            
        Raises:
            MusicGenerationError: If conversion fails
        """
        try:
            # Determine output format from file extension if not specified
            if output_format is None:
                output_format = os.path.splitext(output_file)[1].lower().lstrip(".")
                if not output_format:
                    raise MusicGenerationError("Output format not specified and could not be determined from file extension")
            
            # Check if output format is supported
            supported_formats = ["wav", "flac", "ogg", "mp3"]
            if output_format not in supported_formats:
                raise MusicGenerationError(f"Unsupported output format: {output_format}")
            
            # Read input file
            import soundfile as sf
            data, input_sample_rate = sf.read(input_file)
            
            # Apply conversions
            if sample_rate is not None and sample_rate != input_sample_rate:
                # Resample
                data = self._resample_audio(data, input_sample_rate, sample_rate)
                output_sample_rate = sample_rate
            else:
                output_sample_rate = input_sample_rate
            
            if channels is not None:
                # Convert channels
                data = self._convert_channels(data, channels)
            
            # Determine subtype based on bit depth
            subtype = None
            if bit_depth is not None:
                if bit_depth == 16:
                    subtype = "PCM_16"
                elif bit_depth == 24:
                    subtype = "PCM_24"
                elif bit_depth == 32:
                    subtype = "FLOAT"
            
            # Handle MP3 format separately
            if output_format == "mp3":
                self._save_mp3(data, output_sample_rate, output_file)
            else:
                # Save using soundfile
                sf.write(
                    output_file,
                    data,
                    output_sample_rate,
                    subtype=subtype,
                    format=output_format.upper()
                )
            
            logger.info(f"Converted {input_file} to {output_file}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to convert audio format: {str(e)}")
    
    @with_error_handling
    def convert_sample_rate(
        self, 
        audio: np.ndarray, 
        from_rate: int, 
        to_rate: int
    ) -> np.ndarray:
        """Convert audio sample rate.
        
        Args:
            audio: Audio data as numpy array
            from_rate: Original sample rate
            to_rate: Target sample rate
            
        Returns:
            Resampled audio
            
        Raises:
            MusicGenerationError: If resampling fails
        """
        try:
            return self._resample_audio(audio, from_rate, to_rate)
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to convert sample rate: {str(e)}")
    
    @with_error_handling
    def convert_channels(
        self, 
        audio: np.ndarray, 
        channels: int
    ) -> np.ndarray:
        """Convert audio channel count.
        
        Args:
            audio: Audio data as numpy array
            channels: Target channel count
            
        Returns:
            Audio with converted channel count
            
        Raises:
            MusicGenerationError: If channel conversion fails
        """
        try:
            return self._convert_channels(audio, channels)
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to convert channels: {str(e)}")
    
    def _resample_audio(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio from one sample rate to another.
        
        Args:
            audio: Audio data as numpy array
            from_rate: Original sample rate
            to_rate: Target sample rate
            
        Returns:
            Resampled audio
        """
        if from_rate == to_rate:
            return audio
        
        try:
            # Use scipy for resampling
            from scipy import signal
            
            # Calculate resampling ratio
            ratio = to_rate / from_rate
            
            # Handle multi-channel audio
            if len(audio.shape) > 1:
                # Process each channel separately
                resampled_channels = []
                for i in range(audio.shape[1]):
                    channel = audio[:, i]
                    resampled = signal.resample(channel, int(len(channel) * ratio))
                    resampled_channels.append(resampled)
                
                # Stack channels
                return np.column_stack(resampled_channels)
            else:
                # Mono audio
                return signal.resample(audio, int(len(audio) * ratio))
        
        except Exception as e:
            logger.error(f"Resampling failed: {str(e)}")
            # Return original audio if resampling fails
            return audio
    
    def _convert_channels(self, audio: np.ndarray, channels: int) -> np.ndarray:
        """Convert audio channel count.
        
        Args:
            audio: Audio data as numpy array
            channels: Target channel count
            
        Returns:
            Audio with converted channel count
        """
        # Check current channel count
        if len(audio.shape) == 1:
            current_channels = 1
        else:
            current_channels = audio.shape[1]
        
        # No conversion needed
        if current_channels == channels:
            return audio
        
        # Mono to stereo
        if current_channels == 1 and channels == 2:
            if len(audio.shape) == 1:
                return np.column_stack((audio, audio))
            else:
                return np.column_stack((audio[:, 0], audio[:, 0]))
        
        # Stereo to mono
        if current_channels == 2 and channels == 1:
            return np.mean(audio, axis=1)
        
        # Other conversions not supported
        logger.warning(f"Unsupported channel conversion: {current_channels} to {channels}")
        return audio
    
    def _save_mp3(self, audio: np.ndarray, sample_rate: int, file_path: str) -> None:
        """Save audio as MP3 file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            file_path: Path to save the MP3 file
            
        Raises:
            MusicGenerationError: If MP3 saving fails
        """
        try:
            # Try to import lameenc
            try:
                import lameenc
            except ImportError:
                raise MusicGenerationError("lameenc module not found. Install with 'pip install lameenc' to save MP3 files.")
            
            # Ensure audio is in the right format
            if len(audio.shape) == 1:
                # Mono to stereo
                audio = np.column_stack((audio, audio))
            elif audio.shape[1] > 2:
                # More than 2 channels, convert to stereo
                audio = audio[:, :2]
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create encoder
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(192)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(2)  # Always use stereo for MP3
            encoder.set_quality(2)  # 2=high, 7=fastest
            
            # Encode
            mp3_data = encoder.encode(audio_int16[:, 0], audio_int16[:, 1])
            mp3_data += encoder.flush()
            
            # Write to file
            with open(file_path, "wb") as f:
                f.write(mp3_data)
        
        except Exception as e:
            # Try fallback method with pydub
            try:
                import pydub
                import os
                import soundfile as sf
                
                # Save as WAV first
                temp_wav = file_path + ".temp.wav"
                sf.write(temp_wav, audio, sample_rate, subtype="PCM_16")
                
                # Convert to MP3
                audio_segment = pydub.AudioSegment.from_wav(temp_wav)
                audio_segment.export(file_path, format="mp3", bitrate="192k")
                
                # Remove temporary WAV file
                os.remove(temp_wav)
            
            except ImportError:
                raise MusicGenerationError("Neither lameenc nor pydub module found. Install one of them to save MP3 files.")
            except Exception as e2:
                raise MusicGenerationError(f"Failed to save MP3 file: {str(e2)}")