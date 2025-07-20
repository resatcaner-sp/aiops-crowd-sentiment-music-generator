"""Error handling utilities."""

import logging
from typing import Callable

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError

logger = logging.getLogger(__name__)


def handle_audio_error(error: AudioProcessingError) -> None:
    """Handle audio processing errors.
    
    Args:
        error: The audio processing error
    """
    logger.error(f"Audio processing error: {error.message}")
    # Switch to fallback mode
    # Notify monitoring system


def handle_sync_error(error: SynchronizationError) -> None:
    """Handle synchronization errors.
    
    Args:
        error: The synchronization error
    """
    logger.error(f"Synchronization error: {error.message}")
    # Re-establish synchronization
    # Use best-effort timestamp matching


def handle_music_error(error: MusicGenerationError) -> None:
    """Handle music generation errors.
    
    Args:
        error: The music generation error
    """
    logger.error(f"Music generation error: {error.message}")
    # Switch to pre-composed segments
    # Gradually recover when possible


def with_error_handling(func: Callable) -> Callable:
    """Decorator for handling errors in functions.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AudioProcessingError as e:
            handle_audio_error(e)
        except SynchronizationError as e:
            handle_sync_error(e)
        except MusicGenerationError as e:
            handle_music_error(e)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    return wrapper