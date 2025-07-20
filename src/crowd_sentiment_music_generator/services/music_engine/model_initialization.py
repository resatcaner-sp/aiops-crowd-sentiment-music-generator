"""Model initialization module for Magenta music engine.

This module provides functionality for initializing and configuring Magenta models
for real-time music generation.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple

from crowd_sentiment_music_generator.exceptions.music_generation_error import MusicGenerationError
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class ModelInitializer:
    """Handles initialization and configuration of Magenta models.
    
    This class provides methods for downloading, configuring, and initializing
    Magenta models for real-time music generation.
    """
    
    # Model URLs for downloading
    MODEL_URLS = {
        "performance_rnn": "https://storage.googleapis.com/magentadata/models/performance_rnn/bundle/performance_with_dynamics.mag",
        "performance_rnn_conditional": "https://storage.googleapis.com/magentadata/models/performance_rnn/bundle/performance_with_dynamics_and_modulations.mag",
        "melody_rnn": "https://storage.googleapis.com/magentadata/models/melody_rnn/bundle/attention_rnn.mag"
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the model initializer.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.models_path = self.config.models_path
        logger.info("Initialized ModelInitializer")
    
    @with_error_handling
    def ensure_model_available(self, model_type: str) -> str:
        """Ensure that the specified model is available, downloading if necessary.
        
        Args:
            model_type: Type of model to ensure availability
            
        Returns:
            Path to the model bundle file
            
        Raises:
            MusicGenerationError: If model download or verification fails
        """
        if model_type not in self.MODEL_URLS:
            raise MusicGenerationError(f"Unknown model type: {model_type}")
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_path, exist_ok=True)
            
            # Get model filename from URL
            model_url = self.MODEL_URLS[model_type]
            model_filename = os.path.basename(model_url)
            model_path = os.path.join(self.models_path, model_filename)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.info(f"Model file not found: {model_path}. Downloading...")
                self._download_model(model_url, model_path)
            
            # Verify model file
            if not self._verify_model_file(model_path):
                logger.warning(f"Model file verification failed: {model_path}. Re-downloading...")
                self._download_model(model_url, model_path)
                
                # Verify again
                if not self._verify_model_file(model_path):
                    raise MusicGenerationError(f"Model file verification failed after re-download: {model_path}")
            
            logger.info(f"Model available: {model_type} at {model_path}")
            return model_path
        
        except Exception as e:
            if not isinstance(e, MusicGenerationError):
                raise MusicGenerationError(f"Failed to ensure model availability: {str(e)}")
            raise
    
    def _download_model(self, url: str, destination: str) -> None:
        """Download a model from the specified URL.
        
        Args:
            url: URL to download from
            destination: Destination file path
            
        Raises:
            MusicGenerationError: If download fails
        """
        try:
            import requests
            
            # Download with progress reporting
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(destination, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress
                            if total_size > 0:
                                percent = int(100 * downloaded / total_size)
                                if percent % 10 == 0:
                                    logger.info(f"Download progress: {percent}%")
            
            logger.info(f"Downloaded model to {destination}")
        
        except Exception as e:
            # Clean up partial download
            if os.path.exists(destination):
                os.remove(destination)
            
            raise MusicGenerationError(f"Failed to download model: {str(e)}")
    
    def _verify_model_file(self, model_path: str) -> bool:
        """Verify that the model file is valid.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if the model file is valid, False otherwise
        """
        # Check file size
        try:
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Arbitrary minimum size
                logger.warning(f"Model file too small: {model_path} ({file_size} bytes)")
                return False
            
            # For a more thorough check, we could try to load the model
            # but that would require importing Magenta here
            
            return True
        
        except Exception as e:
            logger.warning(f"Failed to verify model file: {str(e)}")
            return False
    
    @with_error_handling
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for the specified model type.
        
        Args:
            model_type: Type of model to get configuration for
            
        Returns:
            Dictionary with model configuration
            
        Raises:
            MusicGenerationError: If configuration retrieval fails
        """
        try:
            # Default configurations for different model types
            configs = {
                "performance_rnn": {
                    "temperature": 1.0,
                    "steps_per_quarter": 4,
                    "qpm": 120.0,
                    "num_velocity_bins": 32,
                    "instrument": 0,  # Piano
                    "program": 0  # Piano
                },
                "performance_rnn_conditional": {
                    "temperature": 1.0,
                    "steps_per_quarter": 4,
                    "qpm": 120.0,
                    "num_velocity_bins": 32,
                    "instrument": 0,  # Piano
                    "program": 0,  # Piano
                    "condition_on_key": True,
                    "condition_on_dynamics": True
                },
                "melody_rnn": {
                    "temperature": 1.0,
                    "steps_per_quarter": 4,
                    "qpm": 120.0,
                    "instrument": 0,  # Piano
                    "program": 0  # Piano
                }
            }
            
            if model_type not in configs:
                raise MusicGenerationError(f"Unknown model type: {model_type}")
            
            return configs[model_type]
        
        except Exception as e:
            if not isinstance(e, MusicGenerationError):
                raise MusicGenerationError(f"Failed to get model configuration: {str(e)}")
            raise
    
    @with_error_handling
    def create_base_melody(self, melody_type: str, key: str = "C") -> List[int]:
        """Create a base melody of the specified type and key.
        
        Args:
            melody_type: Type of melody (neutral, exciting, tense, sad)
            key: Musical key
            
        Returns:
            List of MIDI note numbers
            
        Raises:
            MusicGenerationError: If melody creation fails
        """
        try:
            # Base melodies in C
            base_melodies = {
                "neutral": [
                    [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
                    [69, 67, 65, 64, 62, 60, 59, 60]   # Descending melody
                ],
                "exciting": [
                    [60, 64, 67, 72, 67, 64, 60],      # C major arpeggio
                    [60, 62, 64, 67, 69, 71, 72]       # Ascending scale
                ],
                "tense": [
                    [60, 63, 66, 69, 66, 63, 60],      # C minor arpeggio
                    [60, 61, 63, 66, 68, 70, 72]       # Minor scale
                ],
                "sad": [
                    [69, 67, 65, 64, 62, 60],          # Descending melody
                    [60, 63, 65, 67, 63, 60]           # Minor pattern
                ]
            }
            
            if melody_type not in base_melodies:
                melody_type = "neutral"
            
            # Select a random melody from the type
            import random
            melody = random.choice(base_melodies[melody_type])
            
            # Transpose to the specified key if not C
            if key != "C":
                # Calculate semitone offset
                key_offsets = {
                    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
                }
                
                # Extract root note from key (e.g., "C Major" -> "C")
                root = key.split()[0]
                
                if root in key_offsets:
                    offset = key_offsets[root]
                    melody = [note + offset for note in melody]
            
            return melody
        
        except Exception as e:
            raise MusicGenerationError(f"Failed to create base melody: {str(e)}")