"""Crowd emotion data model."""

from typing import Dict

from pydantic import BaseModel


class CrowdEmotion(BaseModel):
    """Pydantic model for crowd emotion analysis results.
    
    Attributes:
        emotion: Detected emotion (excitement, joy, tension, disappointment, anger, anticipation, neutral)
        intensity: Intensity level on a scale of 0-100
        confidence: Confidence level of the emotion classification (0-1)
        timestamp: Time when the emotion was detected
        audio_features: Audio features used for emotion classification
    """
    
    emotion: str  # excitement, joy, tension, disappointment, anger, anticipation, neutral
    intensity: float  # 0-100 scale
    confidence: float  # 0-1 scale
    timestamp: float
    audio_features: Dict[str, float]