"""System configuration data model."""

from pydantic import BaseModel


class SystemConfig(BaseModel):
    """Pydantic model for system configuration.
    
    Attributes:
        audio_sample_rate: Sample rate for audio processing
        buffer_size: Size of the buffer for storing audio segments (in seconds)
        update_interval: Interval for updating the system state (in seconds)
        emotion_update_interval: Interval for updating emotion classification (in seconds)
        music_update_interval: Interval for updating music generation (in seconds)
        models_path: Path to the pre-trained models
        cultural_adaptation: Cultural adaptation setting (global or specific region)
    """
    
    audio_sample_rate: int = 22050
    buffer_size: int = 30  # seconds
    update_interval: float = 0.5  # seconds
    emotion_update_interval: float = 2.0  # seconds
    music_update_interval: float = 0.5  # seconds
    models_path: str = "./models"
    cultural_adaptation: str = "global"  # or specific region