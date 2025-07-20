"""Audio output pipeline module.

This module provides functionality for real-time audio rendering, buffer management,
and format conversion for the music generation engine.
"""

import logging
import threading
import time
import io
import os
from typing import Dict, Any, Optional, List, Tuple, BinaryIO, Callable

import numpy as np
import soundfile as sf

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class AudioOutputPipeline:
    """Manages audio output for the music generation engine.
    
    This class provides methods for real-time audio rendering, buffer management,
    and format conversion for the music generation engine.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {
        "wav": {
            "extension": ".wav",
            "subtype": "PCM_16",
            "description": "WAV (16-bit PCM)"
        },
        "flac": {
            "extension": ".flac",
            "subtype": "PCM_24",
            "description": "FLAC (24-bit PCM)"
        },
        "ogg": {
            "extension": ".ogg",
            "subtype": "VORBIS",
            "description": "OGG Vorbis"
        },
        "mp3": {
            "extension": ".mp3",
            "subtype": None,  # Handled separately
            "description": "MP3"
        }
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the audio output pipeline.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.sample_rate = 44100  # Default sample rate
        self.channels = 2  # Stereo output
        self.buffer_size = int(self.sample_rate * self.config.buffer_size)  # Buffer size in samples
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        self.buffer_position = 0
        self.is_recording = False
        self.recording_thread = None
        self.recording_buffer = np.zeros(0)
        self.output_callback = None
        logger.info("Initialized AudioOutputPipeline")
    
    @with_error_handling
    def process_audio_frame(self, frame: np.ndarray, sample_rate: int) -> None:
        """Process an audio frame and add it to the buffer.
        
        Args:
            frame: Audio frame as numpy array
            sample_rate: Sample rate of the audio frame
            
        Raises:
            MusicGenerationError: If audio processing fails
        """
        try:
            # Resample if needed
            if sample_rate != self.sample_rate:
                frame = self._resample_audio(frame, sample_rate, self.sample_rate)
            
            # Convert to stereo if needed
            if len(frame.shape) == 1:
                frame = np.column_stack((frame, frame))
            elif frame.shape[1] == 1:
                frame = np.column_stack((frame, frame))
            
            # Ensure frame is not too large for buffer
            if len(frame) > self.buffer_size:
                frame = frame[:self.buffer_size]
            
            # Add to buffer with circular wrapping
            end_pos = self.buffer_position + len(frame)
            
            if end_pos <= self.buffer_size:
                # Frame fits within remaining buffer
                self.audio_buffer[self.buffer_position:end_pos] = frame
            else:
                # Frame wraps around buffer
                first_part = self.buffer_size - self.buffer_position
                second_part = len(frame) - first_part
                
                self.audio_buffer[self.buffer_position:] = frame[:first_part]
                self.audio_buffer[:second_part] = frame[first_part:]
            
            # Update buffer position
            self.buffer_position = (self.buffer_position + len(frame)) % self.buffer_size
            
            # Call output callback if set
            if self.output_callback:
                self.output_callback(frame, self.sample_rate)
            
            # Add to recording buffer if recording
            if self.is_recording:
                with threading.Lock():
                    self.recording_buffer = np.vstack((self.recording_buffer, frame)) if len(self.recording_buffer) > 0 else frame
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to process audio frame: {str(e)}")
    
    @with_error_handling
    def set_output_callback(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """Set a callback function for audio output.
        
        Args:
            callback: Function that accepts audio frame and sample rate
            
        Raises:
            MusicGenerationError: If callback setting fails
        """
        try:
            self.output_callback = callback
            logger.debug("Set output callback")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to set output callback: {str(e)}")
    
    @with_error_handling
    def get_current_buffer(self) -> Tuple[np.ndarray, int]:
        """Get the current audio buffer.
        
        Returns:
            Tuple of (audio_buffer, sample_rate)
            
        Raises:
            MusicGenerationError: If buffer retrieval fails
        """
        try:
            # Create a copy of the buffer in correct playback order
            buffer_copy = np.zeros_like(self.audio_buffer)
            buffer_copy[:self.buffer_size-self.buffer_position] = self.audio_buffer[self.buffer_position:]
            buffer_copy[self.buffer_size-self.buffer_position:] = self.audio_buffer[:self.buffer_position]
            
            return buffer_copy, self.sample_rate
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to get current buffer: {str(e)}")
    
    @with_error_handling
    def start_recording(self) -> None:
        """Start recording audio output.
        
        Raises:
            MusicGenerationError: If recording start fails
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        try:
            # Reset recording buffer
            self.recording_buffer = np.zeros((0, self.channels))
            
            # Start recording
            self.is_recording = True
            
            logger.info("Started recording")
        
        except Exception as e:
            self.is_recording = False
            raise MusicGenerationError(f"Failed to start recording: {str(e)}")
    
    @with_error_handling
    def stop_recording(self) -> Tuple[np.ndarray, int]:
        """Stop recording and return the recorded audio.
        
        Returns:
            Tuple of (recorded_audio, sample_rate)
            
        Raises:
            MusicGenerationError: If recording stop fails
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return np.zeros((0, self.channels)), self.sample_rate
        
        try:
            # Stop recording
            self.is_recording = False
            
            # Get a copy of the recording buffer
            with threading.Lock():
                recording = self.recording_buffer.copy()
            
            logger.info(f"Stopped recording, {len(recording)} samples recorded")
            
            return recording, self.sample_rate
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to stop recording: {str(e)}")
    
    @with_error_handling
    def save_audio(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        file_path: str, 
        format: str = "wav"
    ) -> None:
        """Save audio to a file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            file_path: Path to save the file
            format: Audio format (wav, flac, ogg, mp3)
            
        Raises:
            MusicGenerationError: If audio saving fails
        """
        try:
            # Check if format is supported
            if format not in self.SUPPORTED_FORMATS:
                raise MusicGenerationError(f"Unsupported audio format: {format}")
            
            # Get format info
            format_info = self.SUPPORTED_FORMATS[format]
            
            # Ensure file has correct extension
            if not file_path.endswith(format_info["extension"]):
                file_path += format_info["extension"]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Handle MP3 format separately
            if format == "mp3":
                self._save_mp3(audio, sample_rate, file_path)
            else:
                # Save using soundfile
                sf.write(
                    file_path,
                    audio,
                    sample_rate,
                    subtype=format_info["subtype"],
                    format=format.upper()
                )
            
            logger.info(f"Saved audio to {file_path}")
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to save audio: {str(e)}")
    
    @with_error_handling
    def get_audio_as_bytes(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        format: str = "wav"
    ) -> bytes:
        """Get audio as bytes in the specified format.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            format: Audio format (wav, flac, ogg, mp3)
            
        Returns:
            Audio data as bytes
            
        Raises:
            MusicGenerationError: If audio conversion fails
        """
        try:
            # Check if format is supported
            if format not in self.SUPPORTED_FORMATS:
                raise MusicGenerationError(f"Unsupported audio format: {format}")
            
            # Get format info
            format_info = self.SUPPORTED_FORMATS[format]
            
            # Handle MP3 format separately
            if format == "mp3":
                return self._get_mp3_bytes(audio, sample_rate)
            else:
                # Use in-memory file
                buffer = io.BytesIO()
                
                # Save to buffer
                sf.write(
                    buffer,
                    audio,
                    sample_rate,
                    subtype=format_info["subtype"],
                    format=format.upper()
                )
                
                # Get bytes
                buffer.seek(0)
                return buffer.read()
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to get audio as bytes: {str(e)}")
    
    @with_error_handling
    def mix_audio_streams(
        self, 
        streams: List[Tuple[np.ndarray, int]], 
        weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, int]:
        """Mix multiple audio streams together.
        
        Args:
            streams: List of (audio, sample_rate) tuples
            weights: Optional list of weights for each stream (default: equal weights)
            
        Returns:
            Tuple of (mixed_audio, sample_rate)
            
        Raises:
            MusicGenerationError: If audio mixing fails
        """
        try:
            if not streams:
                return np.zeros((0, self.channels)), self.sample_rate
            
            # Use equal weights if not provided
            if weights is None:
                weights = [1.0 / len(streams)] * len(streams)
            
            # Ensure weights sum to 1.0
            weight_sum = sum(weights)
            if weight_sum != 1.0:
                weights = [w / weight_sum for w in weights]
            
            # Resample all streams to the same sample rate
            target_rate = self.sample_rate
            resampled_streams = []
            
            for (audio, rate), weight in zip(streams, weights):
                # Resample if needed
                if rate != target_rate:
                    audio = self._resample_audio(audio, rate, target_rate)
                
                # Convert to stereo if needed
                if len(audio.shape) == 1:
                    audio = np.column_stack((audio, audio))
                elif audio.shape[1] == 1:
                    audio = np.column_stack((audio, audio))
                
                # Apply weight
                audio = audio * weight
                
                resampled_streams.append(audio)
            
            # Find the maximum length
            max_length = max(len(audio) for audio in resampled_streams)
            
            # Pad shorter streams
            padded_streams = []
            for audio in resampled_streams:
                if len(audio) < max_length:
                    padding = np.zeros((max_length - len(audio), self.channels))
                    audio = np.vstack((audio, padding))
                padded_streams.append(audio)
            
            # Mix streams
            mixed = np.zeros((max_length, self.channels))
            for audio in padded_streams:
                mixed += audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
            
            return mixed, target_rate
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to mix audio streams: {str(e)}")
    
    @with_error_handling
    def apply_fade(
        self, 
        audio: np.ndarray, 
        fade_in: float = 0.0, 
        fade_out: float = 0.0, 
        sample_rate: int = None
    ) -> np.ndarray:
        """Apply fade-in and fade-out to audio.
        
        Args:
            audio: Audio data as numpy array
            fade_in: Fade-in duration in seconds
            fade_out: Fade-out duration in seconds
            sample_rate: Sample rate of the audio (default: self.sample_rate)
            
        Returns:
            Audio with fades applied
            
        Raises:
            MusicGenerationError: If fade application fails
        """
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Create a copy of the audio
            result = audio.copy()
            
            # Apply fade-in
            if fade_in > 0:
                fade_in_samples = int(fade_in * sample_rate)
                if fade_in_samples > 0 and fade_in_samples < len(result):
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                    
                    # Apply to each channel
                    if len(result.shape) == 1:
                        result[:fade_in_samples] *= fade_in_curve
                    else:
                        for i in range(result.shape[1]):
                            result[:fade_in_samples, i] *= fade_in_curve
            
            # Apply fade-out
            if fade_out > 0:
                fade_out_samples = int(fade_out * sample_rate)
                if fade_out_samples > 0 and fade_out_samples < len(result):
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                    
                    # Apply to each channel
                    if len(result.shape) == 1:
                        result[-fade_out_samples:] *= fade_out_curve
                    else:
                        for i in range(result.shape[1]):
                            result[-fade_out_samples:, i] *= fade_out_curve
            
            return result
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to apply fade: {str(e)}")
    
    @with_error_handling
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to the target level.
        
        Args:
            audio: Audio data as numpy array
            target_level: Target peak level (0-1)
            
        Returns:
            Normalized audio
            
        Raises:
            MusicGenerationError: If normalization fails
        """
        try:
            # Find the maximum absolute value
            max_val = np.max(np.abs(audio))
            
            # Avoid division by zero
            if max_val == 0:
                return audio
            
            # Scale to target level
            return audio * (target_level / max_val)
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to normalize audio: {str(e)}")
    
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
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create encoder
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(192)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(self.channels)
            encoder.set_quality(2)  # 2=high, 7=fastest
            
            # Encode
            if self.channels == 2:
                mp3_data = encoder.encode(audio_int16[:, 0], audio_int16[:, 1])
            else:
                mp3_data = encoder.encode(audio_int16)
            
            mp3_data += encoder.flush()
            
            # Write to file
            with open(file_path, "wb") as f:
                f.write(mp3_data)
        
        except Exception as e:
            # Try fallback method with pydub
            try:
                import pydub
                
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
    
    def _get_mp3_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Get audio as MP3 bytes.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            MP3 audio data as bytes
            
        Raises:
            MusicGenerationError: If MP3 conversion fails
        """
        try:
            # Try to import lameenc
            try:
                import lameenc
            except ImportError:
                raise MusicGenerationError("lameenc module not found. Install with 'pip install lameenc' to get MP3 bytes.")
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create encoder
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(192)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(self.channels)
            encoder.set_quality(2)  # 2=high, 7=fastest
            
            # Encode
            if self.channels == 2:
                mp3_data = encoder.encode(audio_int16[:, 0], audio_int16[:, 1])
            else:
                mp3_data = encoder.encode(audio_int16)
            
            mp3_data += encoder.flush()
            
            return mp3_data
        
        except Exception as e:
            # Try fallback method with pydub
            try:
                import pydub
                import io
                
                # Save as WAV first
                buffer = io.BytesIO()
                sf.write(buffer, audio, sample_rate, subtype="PCM_16", format="WAV")
                buffer.seek(0)
                
                # Convert to MP3
                audio_segment = pydub.AudioSegment.from_wav(buffer)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
                mp3_buffer.seek(0)
                
                return mp3_buffer.read()
            
            except ImportError:
                raise MusicGenerationError("Neither lameenc nor pydub module found. Install one of them to get MP3 bytes.")
            except Exception as e2:
                raise MusicGenerationError(f"Failed to get MP3 bytes: {str(e2)}")