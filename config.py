"""
Configuration file for the Mental Health Screening Application
"""
import os

# Model Configuration
MEDASR_ID = "google/medasr"  # Medical ASR model
# Use MedASR's own processor config to match model inputs.
MEDASR_PROCESSOR_ID = MEDASR_ID

# MedGemma Configuration
# Options: "google/medgemma-2-27b-it" (27B) or "google/medgemma-2-4b-it" (4B)
# Use 4B for production (faster, lower resources), 27B for research (higher accuracy)
MEDGEMMA_MODEL_ID = "google/medgemma-2-4b-it"  # Using 4B model
USE_MEDGEMMA_27B_FOR_CRITICAL = False  # Set to False to always use 4B (no upgrade to 27B)

SR = 16000  # Sample rate for audio
SEED = 42
MAX_NEW_TOKENS = 768  # For ~2-minute clips
MEDGEMMA_MAX_TOKENS = 1024  # Max tokens for MedGemma text generation

# Risk Thresholds
PHQ9_THRESHOLD_MODERATE = 10  # Moderate depression threshold
PHQ9_THRESHOLD_SEVERE = 15   # Severe depression threshold
ANXIETY_THRESHOLD_MODERATE = 0.6  # Moderate anxiety probability
ANXIETY_THRESHOLD_SEVERE = 0.8    # Severe anxiety probability
PTSD_THRESHOLD_MODERATE = 0.6     # Moderate PTSD probability
PTSD_THRESHOLD_SEVERE = 0.8       # Severe PTSD probability

# Critical Alert Thresholds
SUICIDE_RISK_KEYWORDS = [
    "suicide", "kill myself", "end it all", "not worth living",
    "better off dead", "want to die", "no point", "give up"
]
CRITICAL_RISK_THRESHOLD = 0.9  # Probability threshold for critical alerts

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
WAV_CACHE_DIR = os.path.join(BASE_DIR, "wav_cache")
DB_PATH = os.path.join(DATA_DIR, "screening_database.db")

# Database Configuration
DB_TABLE_CALLS = "calls"
DB_TABLE_SCORES = "scores"
DB_TABLE_REPORTS = "reports"
DB_TABLE_ALERTS = "alerts"

# Alert Recipients (to be configured with actual contact info)
ALERT_EMAILS = [
    "medical_officer@military.mil",
    "mental_health_team@military.mil"
]
ALERT_PHONE_NUMBERS = [
    "+1-XXX-XXX-XXXX"  # Crisis hotline
]

# Model Training Configuration
TRAIN_TEST_SPLIT = 0.2
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
LOGISTIC_REGRESSION_MAX_ITER = 3000
CALIBRATION_CV = 3
