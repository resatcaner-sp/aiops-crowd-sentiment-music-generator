"""Emotion classification for crowd analysis.

This module provides functions for classifying crowd emotions based on audio features,
using pre-trained models and a classification pipeline with context-aware analysis.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import pipeline

from crowd_sentiment_music_generator.exceptions.audio_processing_error import AudioProcessingError
from crowd_sentiment_music_generator.models.data.crowd_emotion import CrowdEmotion
from crowd_sentiment_music_generator.models.data.system_config import SystemConfig
from crowd_sentiment_music_generator.utils.error_handlers import with_error_handling

logger = logging.getLogger(__name__)


class EmotionClassifier:
    """Emotion classifier for crowd audio.
    
    This class provides methods for classifying crowd emotions based on audio features,
    using pre-trained models and a multi-modal classification pipeline with context awareness.
    """
    
    # Valid emotion categories
    VALID_EMOTIONS = [
        "excitement", "joy", "tension", "disappointment", 
        "anger", "anticipation", "neutral"
    ]
    
    # Context importance weights for different event types
    CONTEXT_WEIGHTS = {
        "goal": 0.8,
        "red_card": 0.7,
        "yellow_card": 0.5,
        "penalty_awarded": 0.7,
        "near_miss": 0.6,
        "corner": 0.4,
        "free_kick": 0.4,
        "substitution": 0.3,
        "injury": 0.5,
        "offside": 0.3,
        "default": 0.4  # Default weight for unlisted events
    }
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the emotion classifier.
        
        Args:
            config: System configuration (optional, uses default values if not provided)
        """
        self.config = config or SystemConfig()
        self.models = {}
        self.scaler = None
        self.audio_classifier = None
        self._load_models()
        logger.info("Initialized EmotionClassifier with models from %s", self.config.models_path)
    
    def _load_models(self) -> None:
        """Load pre-trained models for emotion classification.
        
        This method loads the following models:
        - Random Forest classifier for feature-based classification
        - Standard scaler for feature normalization
        - Hugging Face audio classification model for additional context
        
        Raises:
            AudioProcessingError: If model loading fails
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.config.models_path, exist_ok=True)
            
            # Path to pre-trained models
            rf_model_path = os.path.join(self.config.models_path, "emotion_rf_model.joblib")
            scaler_path = os.path.join(self.config.models_path, "emotion_scaler.joblib")
            
            # Check if models exist, otherwise create simple placeholder models
            # In a real implementation, these would be properly trained models
            if not os.path.exists(rf_model_path):
                logger.warning("Pre-trained Random Forest model not found, creating placeholder")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                # Train with dummy data to initialize
                X = np.random.rand(100, 20)  # 20 features
                y = np.random.choice(self.VALID_EMOTIONS, 100)
                rf_model.fit(X, y)
                joblib.dump(rf_model, rf_model_path)
            
            if not os.path.exists(scaler_path):
                logger.warning("Feature scaler not found, creating placeholder")
                scaler = StandardScaler()
                # Fit with dummy data to initialize
                X = np.random.rand(100, 20)  # 20 features
                scaler.fit(X)
                joblib.dump(scaler, scaler_path)
            
            # Load models
            self.models["random_forest"] = joblib.load(rf_model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Initialize Hugging Face audio classifier if available
            # This is wrapped in a try-except because it's optional and requires additional dependencies
            try:
                self.audio_classifier = pipeline(
                    "audio-classification",
                    model="MIT/ast-finetuned-audioset-10-10-0.4593"
                )
                logger.info("Loaded Hugging Face audio classifier")
            except Exception as e:
                logger.warning(f"Failed to load Hugging Face audio classifier: {str(e)}")
                logger.warning("Continuing without Hugging Face model")
                self.audio_classifier = None
            
        except Exception as e:
            logger.error(f"Failed to load emotion classification models: {str(e)}")
            raise AudioProcessingError(f"Model loading failed: {str(e)}")
    
    @with_error_handling
    def classify_emotion(
        self, 
        features: Dict[str, float], 
        match_context: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """Classify crowd emotion based on audio features using multi-modal approach.
        
        This method implements a multi-modal classification approach that combines:
        1. Audio feature-based classification using machine learning models
        2. Match context analysis for event-based emotion prediction
        3. Historical trend analysis for temporal consistency
        
        Args:
            features: Dictionary of audio features
            match_context: Optional match context for contextual classification
            
        Returns:
            Tuple of (emotion, confidence)
            
        Raises:
            AudioProcessingError: If classification fails
        """
        # Extract features for classification
        feature_vector = self._prepare_feature_vector(features)
        
        # Apply feature scaling
        scaled_features = self.scaler.transform([feature_vector])
        
        # Get base classification from Random Forest model
        emotion_probabilities = self.models["random_forest"].predict_proba(scaled_features)[0]
        emotion_idx = np.argmax(emotion_probabilities)
        emotion = self.models["random_forest"].classes_[emotion_idx]
        confidence = emotion_probabilities[emotion_idx]
        
        # Calculate audio quality factor for confidence adjustment
        audio_quality = self._calculate_audio_quality_factor(features)
        
        # Adjust base confidence by audio quality
        confidence = confidence * audio_quality
        
        # Apply context-based adjustments if match context is provided
        if match_context:
            emotion, confidence = self._apply_context_adjustments(
                emotion, confidence, features, match_context
            )
            
            # Apply additional confidence scoring based on multi-modal agreement
            confidence = self._calculate_confidence_score(
                emotion, confidence, features, match_context
            )
        
        return emotion, confidence
        
    def _calculate_audio_quality_factor(self, features: Dict[str, float]) -> float:
        """Calculate a quality factor for the audio features.
        
        This method assesses the quality of the audio features to adjust
        confidence scores. Lower quality audio should result in lower confidence.
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Quality factor between 0.5 and 1.0
        """
        # Start with base quality of 1.0
        quality_factor = 1.0
        
        # Check isolation quality if available
        if "isolation_quality" in features:
            # Scale isolation quality to range 0.7-1.0
            # Even with poor isolation, we still have some confidence
            isolation_factor = 0.7 + (0.3 * features["isolation_quality"])
            quality_factor *= isolation_factor
        
        # Check signal-to-noise ratio if available
        if "snr" in features:
            snr = features["snr"]
            # SNR below 10dB is poor, above 20dB is good
            if snr < 10:
                snr_factor = 0.7 + (0.3 * (snr / 10))
            else:
                snr_factor = min(1.0, 0.85 + (0.15 * min(1.0, (snr - 10) / 10)))
            quality_factor *= snr_factor
        
        # Check RMS energy (very low energy might be silence or background noise)
        if "rms_energy" in features:
            rms = features["rms_energy"]
            if rms < 0.1:  # Very low energy
                energy_factor = 0.7 + (3.0 * rms)  # Scales from 0.7 to 1.0 as rms goes from 0 to 0.1
            else:
                energy_factor = 1.0
            quality_factor *= energy_factor
        
        # Ensure quality factor is at least 0.5
        return max(0.5, quality_factor)
    
    def _calculate_confidence_score(
        self,
        emotion: str,
        base_confidence: float,
        features: Dict[str, float],
        match_context: Dict
    ) -> float:
        """Calculate a comprehensive confidence score based on multiple factors.
        
        This method implements a sophisticated confidence scoring system that considers:
        1. Base model confidence
        2. Audio quality
        3. Context consistency
        4. Historical consistency
        5. Multi-modal agreement
        
        Args:
            emotion: Classified emotion
            base_confidence: Initial confidence from classification
            features: Dictionary of audio features
            match_context: Match context information
            
        Returns:
            Adjusted confidence score between 0.0 and 1.0
        """
        # Start with base confidence
        confidence = base_confidence
        
        # Factor 1: Context consistency
        # Check if the emotion is consistent with the match context
        event_type = match_context.get("last_event_type")
        if event_type:
            # Define expected emotions for different event types
            expected_emotions = {
                "goal": ["excitement", "joy", "disappointment"],
                "near_miss": ["anticipation", "tension", "disappointment"],
                "red_card": ["anger", "tension", "excitement"],
                "penalty_awarded": ["tension", "anticipation"],
                "yellow_card": ["tension"],
                "corner": ["anticipation"],
                "free_kick": ["anticipation", "tension"]
            }
            
            # If emotion matches expected emotions for the event, boost confidence
            if event_type in expected_emotions and emotion in expected_emotions[event_type]:
                confidence = min(1.0, confidence * 1.15)
            # If emotion is completely inconsistent, reduce confidence
            elif event_type in expected_emotions and emotion not in expected_emotions[event_type]:
                confidence = confidence * 0.85
        
        # Factor 2: Historical consistency
        # Check if the emotion is consistent with recent emotions
        if "previous_emotions" in match_context:
            prev_emotions = match_context["previous_emotions"]
            if prev_emotions and emotion in prev_emotions:
                # Emotion is consistent with recent history
                confidence = min(1.0, confidence * 1.1)
            elif prev_emotions and emotion not in prev_emotions and len(prev_emotions) >= 3:
                # Emotion is a sudden change from consistent history
                confidence = confidence * 0.9
        
        # Factor 3: Multi-modal agreement
        # If we have predictions from multiple models, check agreement
        if "audio_classifier_prediction" in match_context:
            audio_pred = match_context["audio_classifier_prediction"]
            if audio_pred == emotion:
                # Multiple models agree, boost confidence
                confidence = min(1.0, confidence * 1.2)
            else:
                # Models disagree, reduce confidence
                confidence = confidence * 0.8
        
        # Factor 4: Match importance and time
        # Critical moments in important matches should have higher confidence
        match_importance = match_context.get("match_importance", 0.5)
        time_remaining = match_context.get("time_remaining", 45)
        
        if match_importance > 0.7 and time_remaining < 10:
            # Critical moment in important match
            confidence = min(1.0, confidence * 1.1)
        
        # Factor 5: Crowd energy consistency
        # High energy should correlate with high-arousal emotions
        if "rms_energy" in features:
            high_arousal = emotion in ["excitement", "anger", "tension"]
            high_energy = features["rms_energy"] > 0.5
            
            if (high_arousal and high_energy) or (not high_arousal and not high_energy):
                # Energy level is consistent with emotion arousal
                confidence = min(1.0, confidence * 1.1)
            else:
                # Energy level is inconsistent with emotion arousal
                confidence = confidence * 0.9
        
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector for classification.
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            List of feature values in the expected order
            
        Raises:
            AudioProcessingError: If feature preparation fails
        """
        try:
            # Define the expected features and their order
            expected_features = [
                "rms_energy", "spectral_centroid", "spectral_rolloff",
                "zero_crossing_rate", "tempo", "spectral_contrast"
            ]
            
            # Add MFCC features if available
            mfcc_features = [f"mfcc_{i+1}" for i in range(13)]
            expected_features.extend(mfcc_features)
            
            # Create feature vector with default values of 0 for missing features
            feature_vector = [features.get(feature, 0.0) for feature in expected_features]
            
            return feature_vector
        except Exception as e:
            logger.error(f"Failed to prepare feature vector: {str(e)}")
            raise AudioProcessingError(f"Feature preparation failed: {str(e)}")
    
    def _apply_context_adjustments(
        self, 
        emotion: str, 
        confidence: float, 
        features: Dict[str, float],
        match_context: Dict
    ) -> Tuple[str, float]:
        """Apply context-aware adjustments to emotion classification using a multi-modal approach.
        
        This method integrates match context with audio analysis to provide more accurate
        emotion classification. It uses a multi-modal approach that combines:
        1. Audio feature-based classification
        2. Match event context
        3. Game state information
        4. Historical patterns
        
        Args:
            emotion: Initial emotion classification
            confidence: Initial confidence score
            features: Dictionary of audio features
            match_context: Match context information
            
        Returns:
            Tuple of (adjusted_emotion, adjusted_confidence)
        """
        # Extract relevant context information
        event_type = match_context.get("last_event_type")
        score_diff = match_context.get("score_difference", 0)
        time_remaining = match_context.get("time_remaining", 45)
        home_team = match_context.get("home_team")
        away_team = match_context.get("away_team")
        event_team = match_context.get("event_team")
        is_home_event = event_team == home_team if event_team and home_team else None
        match_importance = match_context.get("match_importance", 0.5)  # 0.0-1.0 scale
        
        # Get context weight based on event type
        context_weight = self.CONTEXT_WEIGHTS.get(event_type, self.CONTEXT_WEIGHTS["default"]) if event_type else 0.0
        
        # Adjust context weight based on match importance and time remaining
        if time_remaining < 10:  # Late game events have higher impact
            context_weight *= 1.2
        if match_importance > 0.7:  # Important matches have higher emotional impact
            context_weight *= 1.1
        
        # Cap context weight at 0.9 to always preserve some audio-based classification
        context_weight = min(0.9, context_weight)
        
        # Get context-based emotion prediction
        context_emotion, context_confidence = self._predict_emotion_from_context(
            event_type, score_diff, time_remaining, is_home_event, match_context
        )
        
        # If we have a strong context-based prediction, blend it with audio-based prediction
        if context_emotion and context_confidence > 0.0:
            # Calculate final emotion and confidence using weighted blend
            if context_emotion == emotion:
                # If both methods agree, increase confidence
                final_confidence = min(1.0, confidence + (context_confidence * 0.2))
                return emotion, final_confidence
            else:
                # If methods disagree, use weighted decision
                audio_weight = 1.0 - context_weight
                
                # Adjust weights based on confidence scores
                audio_weighted_confidence = confidence * audio_weight
                context_weighted_confidence = context_confidence * context_weight
                
                # Choose emotion with higher weighted confidence
                if context_weighted_confidence > audio_weighted_confidence:
                    # Calculate blended confidence score
                    blended_confidence = (context_weighted_confidence + (audio_weighted_confidence * 0.3)) / (context_weight + (audio_weight * 0.3))
                    return context_emotion, blended_confidence
                else:
                    # Keep original emotion but adjust confidence
                    adjusted_confidence = confidence * (1.0 - (context_weight * 0.3))
                    return emotion, adjusted_confidence
        
        # No strong context prediction, return original classification
        return emotion, confidence
    
    def _predict_emotion_from_context(
        self,
        event_type: Optional[str],
        score_diff: int,
        time_remaining: float,
        is_home_event: Optional[bool],
        match_context: Dict
    ) -> Tuple[Optional[str], float]:
        """Predict emotion based solely on match context.
        
        This method implements a rule-based system to predict crowd emotion
        based on match events and state, without using audio features.
        
        Args:
            event_type: Type of the last event
            score_diff: Score difference (positive for home team leading)
            time_remaining: Minutes remaining in the match
            is_home_event: Whether the event was for the home team
            match_context: Full match context dictionary
            
        Returns:
            Tuple of (predicted_emotion, confidence)
        """
        # Default return if no strong prediction
        if not event_type:
            return None, 0.0
        
        # Event-based predictions
        if event_type == "goal":
            # Goal events generate different emotions based on context
            if is_home_event is True:
                # Home team scored
                return "excitement" if score_diff >= 0 else "joy", 0.85
            elif is_home_event is False:
                # Away team scored
                if score_diff > 1:  # Home team still leading by more than 1
                    return "disappointment", 0.6
                else:
                    return "disappointment", 0.8
            else:
                # Unknown team scored
                return "excitement", 0.7
        
        elif event_type == "near_miss":
            # Near miss generates anticipation or disappointment
            if time_remaining < 10:  # Late game
                if is_home_event is True and score_diff <= 0:
                    # Home team near miss when tied or behind
                    return "tension", 0.8
                else:
                    return "anticipation", 0.7
            else:
                return "anticipation", 0.6
        
        elif event_type == "red_card":
            # Red card usually generates tension or anger
            if is_home_event is True:
                # Against home team
                return "anger", 0.75
            else:
                # Against away team
                return "excitement", 0.6
        
        elif event_type == "penalty_awarded":
            # Penalty awarded generates high tension
            if is_home_event is True:
                # For home team
                return "anticipation", 0.85
            else:
                # Against home team
                return "tension", 0.9
        
        elif event_type == "yellow_card":
            # Yellow card has moderate impact
            if time_remaining < 15 and match_context.get("yellow_card_count", {}).get(event_team, 0) > 3:
                # Multiple yellow cards late in game
                return "tension", 0.7
            else:
                return None, 0.0  # Not significant enough
        
        # Game state based predictions (when no specific event)
        elif event_type == "game_state":
            # Close score in late game
            if abs(score_diff) <= 1 and time_remaining < 10:
                return "tension", 0.75
            # Blowout game
            elif abs(score_diff) >= 3:
                if score_diff > 0:
                    return "joy", 0.6
                else:
                    return "disappointment", 0.7
            # Injury time with close score
            elif match_context.get("is_injury_time", False) and abs(score_diff) <= 1:
                return "tension", 0.85
        
        # No strong prediction from context
        return None, 0.0
    
    @with_error_handling
    def measure_intensity(self, features: Dict[str, float], emotion: str) -> float:
        """Measure emotional intensity based on audio features and classified emotion.
        
        Args:
            features: Dictionary of audio features
            emotion: Classified emotion
            
        Returns:
            Intensity score on a scale of 0-100
            
        Raises:
            AudioProcessingError: If intensity measurement fails
        """
        try:
            # Base intensity from RMS energy (0-1 scale)
            base_intensity = features.get("rms_energy", 0.0)
            
            # Adjust based on spectral features
            spectral_factor = features.get("spectral_centroid", 0.0) / 5000.0  # Normalize to ~0-1
            spectral_factor = min(1.0, spectral_factor)  # Cap at 1.0
            
            # Adjust based on zero crossing rate for percussiveness
            zcr_factor = features.get("zero_crossing_rate", 0.0) / 0.2  # Normalize to ~0-1
            zcr_factor = min(1.0, zcr_factor)  # Cap at 1.0
            
            # Emotion-specific adjustments
            emotion_factors = {
                "excitement": 1.2,
                "joy": 1.0,
                "tension": 0.9,
                "disappointment": 0.7,
                "anger": 1.1,
                "anticipation": 0.8,
                "neutral": 0.5
            }
            
            # Calculate weighted intensity
            emotion_factor = emotion_factors.get(emotion, 1.0)
            intensity = (base_intensity * 0.5 + spectral_factor * 0.3 + zcr_factor * 0.2) * emotion_factor
            
            # Scale to 0-100
            intensity = min(100.0, intensity * 100.0)
            
            return intensity
        except Exception as e:
            logger.error(f"Failed to measure intensity: {str(e)}")
            raise AudioProcessingError(f"Intensity measurement failed: {str(e)}")
    
    @with_error_handling
    def create_crowd_emotion(
        self, 
        features: Dict[str, float], 
        timestamp: float,
        match_context: Optional[Dict] = None
    ) -> CrowdEmotion:
        """Create a CrowdEmotion object from audio features.
        
        This method combines emotion classification and intensity measurement
        to create a complete CrowdEmotion object.
        
        Args:
            features: Dictionary of audio features
            timestamp: Time when the audio was recorded
            match_context: Optional match context for contextual classification
            
        Returns:
            CrowdEmotion object
            
        Raises:
            AudioProcessingError: If emotion creation fails
        """
        # Classify emotion
        emotion, confidence = self.classify_emotion(features, match_context)
        
        # Measure intensity
        intensity = self.measure_intensity(features, emotion)
        
        # Create CrowdEmotion object
        crowd_emotion = CrowdEmotion(
            emotion=emotion,
            intensity=intensity,
            confidence=confidence,
            timestamp=timestamp,
            audio_features=features
        )
        
        return crowd_emotion