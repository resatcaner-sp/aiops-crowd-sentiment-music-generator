"""Error handling utilities."""

import functools
import logging
from typing import Any, Callable, TypeVar

from crowd_sentiment_music_generator.exceptions.data_api_error import DataAPIError
from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.exceptions.synchronization_error import SynchronizationError

# Type variable for function return type
T = TypeVar("T")

# Set up logger
logger = logging.getLogger(__name__)


def with_error_handling(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for handling errors in functions.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Wrapper function that handles errors.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The result of the decorated function
            
        Raises:
            DataAPIError: If a data API error occurs
            AudioProcessingError: If an audio processing error occurs
            MusicGenerationError: If a music generation error occurs
            SynchronizationError: If a synchronization error occurs
            Exception: If any other error occurs
        """
        try:
            return func(*args, **kwargs)
        except DataAPIError as e:
            logger.error(f"Data API error in {func.__name__}: {e.message}")
            raise
        except AudioProcessingError as e:
            logger.error(f"Audio processing error in {func.__name__}: {str(e)}")
            raise
        except MusicGenerationError as e:
            logger.error(f"Music generation error in {func.__name__}: {str(e)}")
            raise
        except SynchronizationError as e:
            logger.error(f"Synchronization error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper