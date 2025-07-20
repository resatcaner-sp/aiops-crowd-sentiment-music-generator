"""Crowd analyzer module for audio analysis.

This module provides the main CrowdAnalyzer class that integrates
feature extraction, signal processing, and context-aware analysis to analyze crowd audio.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.services.crowd_analysis.emotion_classification import EmotionClassifier
from crowd_sentiment_music_generator.services.crowd_analysis.feature_extraction import (
    extract_all_features, isolate_crowd_noise
)
from crowd_sentiment_music_generator.services.crowd_analysis.signal_processing import (
    preprocess_audio, segment_audio
)
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class CrowdAnalyzer:
    """Analyzes crowd audio to determine emotional states.
    
    This class provides methods for processing crowd audio, extracting features,
    and classifying crowd emotions with context-aware analysis.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the crowd analyzer.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.emotion_classifier = EmotionClassifier(self.config)
        self.context_history: List[Dict[str, Any]] = []
        self.max_history_size = 10  # Store last 10 context entries
        logger.info("Initialized CrowdAnalyzer with sample rate %d Hz", self.config.audio_sample_rate)
    
    def update_context_history(self, match_context: Dict[str, Any]) -> None:
        """Update the context history with new match context.
        
        This method maintains a history of recent match contexts to enable
        trend analysis and more sophisticated context-aware classification.
        
        Args:
            match_context: Current match context information
        """
        if not match_context:
            return
            
        # Add timestamp if not present
        if "timestamp" not in match_context:
            match_context["timestamp"] = match_context.get("match_time", 0)
            
        # Add to history
        self.context_history.append(match_context)
        
        # Trim history if needed
        if len(self.context_history) > self.max_history_size:
            self.context_history = self.context_history[-self.max_history_size:]
            
        logger.debug("Updated context history, now contains %d entries", len(self.context_history))
    
    def enrich_match_context(self, match_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich match context with historical data and derived metrics.
        
        This method adds additional context information based on historical data
        and calculates derived metrics that can improve emotion classification.
        
        Args:
            match_context: Current match context information (can be None)
            
        Returns:
            Enriched match context dictionary
        """
        if not match_context:
            return {}
            
        enriched_context = match_context.copy()
        
        # Add historical context if available
        if self.context_history:
            # Calculate event frequency (events per minute)
            recent_events = [ctx for ctx in self.context_history 
                            if ctx.get("last_event_type") and 
                            ctx.get("timestamp", 0) > match_context.get("timestamp", 0) - 300]  # Last 5 minutes
            
            if recent_events:
                enriched_context["event_frequency"] = len(recent_events) / 5.0  # Events per minute
            
            # Detect momentum shifts
            if len(self.context_history) >= 3:
                # Check for scoring pattern changes
                recent_goals = [ctx for ctx in self.context_history[-5:] 
                               if ctx.get("last_event_type") == "goal"]
                
                if len(recent_goals) >= 2:
                    # Check if goals were scored by different teams (momentum shift)
                    scoring_teams = [goal.get("event_team") for goal in recent_goals]
                    if len(set(scoring_teams)) > 1:
                        enriched_context["momentum_shift"] = True
            
            # Add trend information
            if len(self.context_history) >= 3:
                # Calculate crowd energy trend
                if all("crowd_energy" in ctx for ctx in self.context_history[-3:]):
                    energy_values = [ctx["crowd_energy"] for ctx in self.context_history[-3:]]
                    energy_trend = energy_values[-1] - energy_values[0]
                    enriched_context["crowd_energy_trend"] = energy_trend
        
        # Add match phase information
        match_time = match_context.get("match_time", 0)
        if match_time < 15:
            enriched_context["match_phase"] = "opening"
        elif match_time > 75:
            enriched_context["match_phase"] = "closing"
        elif 42 <= match_time <= 48:
            enriched_context["match_phase"] = "halftime_transition"
        else:
            enriched_context["match_phase"] = "middle"
            
        return enriched_context
    
    @with_error_handling
    def process_audio_segment(
        self, 
        audio_segment: np.ndarray, 
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process an audio segment to extract crowd noise and features.
        
        Args:
            audio_segment: Audio segment as numpy array
            sr: Sample rate (optional, uses config value if not provided)
            
        Returns:
            Tuple of (isolated crowd noise, audio features)
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        if sr is None:
            sr = self.config.audio_sample_rate
        
        # Check if audio segment is valid
        if audio_segment.size == 0:
            raise AudioProcessingError("Empty audio segment")
        
        # Preprocess audio
        preprocessed_audio = preprocess_audio(audio_segment, sr)
        
        # Isolate crowd noise
        crowd_noise, isolation_quality = isolate_crowd_noise(preprocessed_audio, sr)
        logger.debug("Isolated crowd noise with quality score: %.2f", isolation_quality)
        
        # Extract features
        features = extract_all_features(crowd_noise, sr, self.config)
        
        # Add isolation quality to features
        features["isolation_quality"] = isolation_quality
        
        return crowd_noise, features
    
    @with_error_handling
    def process_audio_stream(
        self, 
        audio_stream: np.ndarray, 
        sr: Optional[int] = None,
        segment_duration: float = 2.0
    ) -> Dict[str, Dict[str, float]]:
        """Process an audio stream by segmenting it and extracting features from each segment.
        
        Args:
            audio_stream: Audio stream as numpy array
            sr: Sample rate (optional, uses config value if not provided)
            segment_duration: Duration of each segment in seconds (default: 2.0)
            
        Returns:
            Dictionary mapping segment index to feature dictionary
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        if sr is None:
            sr = self.config.audio_sample_rate
        
        # Segment audio
        segments = segment_audio(audio_stream, sr, segment_duration)
        logger.debug("Segmented audio stream into %d segments", len(segments))
        
        # Process each segment
        results = {}
        for i, segment in enumerate(segments):
            try:
                _, features = self.process_audio_segment(segment, sr)
                results[f"segment_{i}"] = features
            except AudioProcessingError as e:
                logger.warning(f"Failed to process segment {i}: {e.message}")
                # Continue with next segment
        
        return results
    
    @with_error_handling
    def get_average_features(
        self, 
        segment_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate average features across multiple segments.
        
        Args:
            segment_features: Dictionary mapping segment index to feature dictionary
            
        Returns:
            Dictionary of average features
            
        Raises:
            AudioProcessingError: If feature averaging fails
        """
        if not segment_features:
            raise AudioProcessingError("No segment features provided")
        
        # Get all feature names from the first segment
        first_segment = next(iter(segment_features.values()))
        feature_names = first_segment.keys()
        
        # Calculate average for each feature
        avg_features = {}
        for feature_name in feature_names:
            values = [segment[feature_name] for segment in segment_features.values() 
                     if feature_name in segment]
            if values:
                avg_features[feature_name] = sum(values) / len(values)
        
        return avg_features
    
    @with_error_handling
    def classify_emotion(
        self, 
        features: Dict[str, float], 
        timestamp: float,
        match_context: Optional[Dict] = None
    ) -> CrowdEmotion:
        """Classify crowd emotion based on audio features and match context.
        
        This method implements context-aware analysis by enriching the match context
        with historical data and derived metrics before classification.
        
        Args:
            features: Dictionary of audio features
            timestamp: Time when the audio was recorded
            match_context: Optional match context for contextual classification
            
        Returns:
            CrowdEmotion object with emotion classification results
            
        Raises:
            AudioProcessingError: If emotion classification fails
        """
        # Update context history if context is provided
        if match_context:
            self.update_context_history(match_context)
            
        # Enrich match context with historical data and derived metrics
        enriched_context = self.enrich_match_context(match_context)
        
        # Use the emotion classifier to create a CrowdEmotion object with enriched context
        crowd_emotion = self.emotion_classifier.create_crowd_emotion(
            features, timestamp, enriched_context
        )
        
        # Store the emotion result in the latest context entry if available
        if self.context_history:
            self.context_history[-1]["crowd_emotion"] = crowd_emotion.emotion
            self.context_history[-1]["crowd_intensity"] = crowd_emotion.intensity
            self.context_history[-1]["crowd_confidence"] = crowd_emotion.confidence
            
            # Calculate crowd energy based on emotion and intensity
            energy_map = {
                "excitement": 1.0,
                "joy": 0.8,
                "tension": 0.7,
                "anticipation": 0.6,
                "anger": 0.9,
                "disappointment": 0.4,
                "neutral": 0.3
            }
            emotion_factor = energy_map.get(crowd_emotion.emotion, 0.5)
            crowd_energy = (crowd_emotion.intensity / 100.0) * emotion_factor
            self.context_history[-1]["crowd_energy"] = crowd_energy
        
        logger.debug(
            "Classified crowd emotion: %s (intensity: %.1f, confidence: %.2f)",
            crowd_emotion.emotion, crowd_emotion.intensity, crowd_emotion.confidence
        )
        
        return crowd_emotion
    
    @with_error_handling
    def classify_audio_segment(
        self, 
        audio_segment: np.ndarray, 
        timestamp: float,
        sr: Optional[int] = None,
        match_context: Optional[Dict] = None
    ) -> CrowdEmotion:
        """Process an audio segment and classify the crowd emotion.
        
        This method combines audio processing and emotion classification
        into a single convenient method.
        
        Args:
            audio_segment: Audio segment as numpy array
            timestamp: Time when the audio was recorded
            sr: Sample rate (optional, uses config value if not provided)
            match_context: Optional match context for contextual classification
            
        Returns:
            CrowdEmotion object with emotion classification results
            
        Raises:
            AudioProcessingError: If processing or classification fails
        """
        # Process audio segment to extract features
        _, features = self.process_audio_segment(audio_segment, sr)
        
        # Classify emotion based on features
        return self.classify_emotion(features, timestamp, match_context)
    
    @with_error_handling
    def classify_audio_stream(
        self, 
        audio_stream: np.ndarray, 
        timestamp: float,
        sr: Optional[int] = None,
        segment_duration: float = 2.0,
        match_context: Optional[Dict] = None
    ) -> CrowdEmotion:
        """Process an audio stream and classify the overall crowd emotion.
        
        This method segments the audio stream, processes each segment,
        averages the features, and then classifies the emotion.
        
        Args:
            audio_stream: Audio stream as numpy array
            timestamp: Time when the audio was recorded
            sr: Sample rate (optional, uses config value if not provided)
            segment_duration: Duration of each segment in seconds (default: 2.0)
            match_context: Optional match context for contextual classification
            
        Returns:
            CrowdEmotion object with emotion classification results
            
        Raises:
            AudioProcessingError: If processing or classification fails
        """
        # Process audio stream to get segment features
        segment_features = self.process_audio_stream(audio_stream, sr, segment_duration)
        
        # Calculate average features across segments
        avg_features = self.get_average_features(segment_features)
        
        # Classify emotion based on average features
        return self.classify_emotion(avg_features, timestamp, match_context)