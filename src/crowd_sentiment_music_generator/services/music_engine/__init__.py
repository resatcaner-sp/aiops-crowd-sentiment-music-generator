"""Music engine service package.

This package provides the music generation engine for the crowd sentiment music generator.
"""

from crowd_sentiment_music_generator.services.music_engine.magenta_engine import MagentaMusicEngine
from crowd_sentiment_music_generator.services.music_engine.model_initialization import ModelInitializer
from crowd_sentiment_music_generator.services.music_engine.continuous_evolution import ContinuousMusicEvolution, TransitionManager
from crowd_sentiment_music_generator.services.music_engine.accent_triggering import AccentTrigger
from crowd_sentiment_music_generator.services.music_engine.audio_pipeline import AudioOutputPipeline
from crowd_sentiment_music_generator.services.music_engine.streaming_output import StreamingAudioOutput, AudioFormatConverter

__all__ = [
    "MagentaMusicEngine",
    "ModelInitializer",
    "ContinuousMusicEvolution",
    "TransitionManager",
    "AccentTrigger",
    "AudioOutputPipeline",
    "StreamingAudioOutput",
    "AudioFormatConverter"
]