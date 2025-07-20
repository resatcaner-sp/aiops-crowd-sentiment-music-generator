"""Custom exceptions for crowd sentiment music generator."""

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError
from crowd_sentiment_music_generator.exceptions.data_api_error import DataAPIError

__all__ = [
    "AudioProcessingError",
    "MusicGenerationError",
    "SynchronizationError",
    "DataAPIError",
]