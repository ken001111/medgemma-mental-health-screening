"""
Mental health classifiers for PHQ-9 (depression), anxiety, and PTSD.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Tuple
import pickle
import os
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    LOGISTIC_REGRESSION_MAX_ITER, CALIBRATION_CV, ARTIFACTS_DIR
)


class MentalHealthClassifier:
    """
    Base class for mental health classifiers.
    Supports multimodal features (text + prosody).
    """
    
    def __init__(self, condition_name: str):
        """
        Initialize classifier.
        
        Args:
            condition_name: Name of the condition (e.g., 'phq9', 'anxiety', 'ptsd')
        """
        self.condition_name = condition_name
        self.text_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF
        )
        self.classifier = None
        self.is_fitted = False
    
    def _extract_text_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from text."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction.")
        return self.text_vectorizer.transform(texts)
    
    def _extract_prosody_features(self, prosody_list: List[Dict[str, float]]) -> np.ndarray:
        """Extract prosody features into a numpy array."""
        if not prosody_list:
            return np.zeros((1, 9))  # Default 9 prosody features
        
        feature_names = [
            "duration", "mean_pitch", "pitch_std", "speaking_rate",
            "mean_energy", "energy_std", "silence_ratio",
            "mean_spectral_centroid", "jitter"
        ]
        
        features = []
        for prosody in prosody_list:
            feature_vector = [prosody.get(name, 0.0) for name in feature_names]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _combine_features(self, text_features: np.ndarray, prosody_features: np.ndarray) -> np.ndarray:
        """Combine text and prosody features."""
        # Normalize prosody features
        prosody_normalized = (prosody_features - prosody_features.mean(axis=0)) / (prosody_features.std(axis=0) + 1e-6)
        
        # Combine using feature concatenation
        # For sparse text features, convert to dense for combination
        if hasattr(text_features, 'toarray'):
            text_features = text_features.toarray()
        
        return np.hstack([text_features, prosody_normalized])
    
    def fit(self, texts: List[str], prosody_list: List[Dict[str, float]], labels: np.ndarray):
        """
        Train the classifier.
        
        Args:
            texts: List of transcribed texts
            prosody_list: List of prosody feature dictionaries
            labels: Binary labels (0/1) or continuous scores
        """
        # Extract features
        text_features = self.text_vectorizer.fit_transform(texts)
        prosody_features = self._extract_prosody_features(prosody_list)
        
        # Combine features
        combined_features = self._combine_features(text_features, prosody_features)
        
        # Train classifier
        base_clf = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            n_jobs=-1
        )
        self.classifier = CalibratedClassifierCV(
            base_clf,
            method="sigmoid",
            cv=CALIBRATION_CV
        )
        self.classifier.fit(combined_features, labels)
        self.is_fitted = True
    
    def predict_proba(self, texts: List[str], prosody_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            texts: List of transcribed texts
            prosody_list: List of prosody feature dictionaries
        
        Returns:
            Probability array (n_samples, 2) for binary classification
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction.")
        
        text_features = self._extract_text_features(texts)
        prosody_features = self._extract_prosody_features(prosody_list)
        combined_features = self._combine_features(text_features, prosody_features)
        
        return self.classifier.predict_proba(combined_features)
    
    def predict_score(self, texts: List[str], prosody_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Predict continuous scores (for PHQ-9 regression).
        
        Args:
            texts: List of transcribed texts
            prosody_list: List of prosody feature dictionaries
        
        Returns:
            Predicted scores
        """
        # For now, use probability as proxy for score
        # In production, you'd train a regression model
        proba = self.predict_proba(texts, prosody_list)
        return proba[:, 1] * 27  # Scale to 0-27 for PHQ-9
    
    def save(self, filepath: str):
        """Save the classifier to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'text_vectorizer': self.text_vectorizer,
                'classifier': self.classifier,
                'is_fitted': self.is_fitted,
                'condition_name': self.condition_name
            }, f)
    
    def load(self, filepath: str):
        """Load the classifier from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.text_vectorizer = data['text_vectorizer']
            self.classifier = data['classifier']
            self.is_fitted = data['is_fitted']
            self.condition_name = data['condition_name']


class PHQ9Classifier(MentalHealthClassifier):
    """PHQ-9 Depression Screening Classifier."""
    
    def __init__(self):
        super().__init__("phq9")
    
    def predict_phq9_score(self, texts: List[str], prosody_list: List[Dict[str, float]]) -> float:
        """Predict PHQ-9 score (0-27)."""
        scores = self.predict_score(texts, prosody_list)
        return float(scores[0])


class AnxietyClassifier(MentalHealthClassifier):
    """Anxiety Screening Classifier."""
    
    def __init__(self):
        super().__init__("anxiety")
    
    def predict_anxiety_risk(self, texts: List[str], prosody_list: List[Dict[str, float]]) -> float:
        """Predict anxiety risk probability (0-1)."""
        proba = self.predict_proba(texts, prosody_list)
        return float(proba[0, 1])


class PTSDClassifier(MentalHealthClassifier):
    """PTSD Screening Classifier."""
    
    def __init__(self):
        super().__init__("ptsd")
    
    def predict_ptsd_risk(self, texts: List[str], prosody_list: List[Dict[str, float]]) -> float:
        """Predict PTSD risk probability (0-1)."""
        proba = self.predict_proba(texts, prosody_list)
        return float(proba[0, 1])
