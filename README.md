# Mental Health Screening Application for Soldiers

A comprehensive multimodal mental health screening system that analyzes 2-minute phone calls using Google's MedASR (speech recognition) and MedGemma (medical AI) to assess depression (PHQ-9), anxiety, and PTSD risk levels. Features automated alerting for high-risk cases and medical report generation for clinician review.

## Features

1. **Phone Call Processing**: Accepts 2-minute phone call recordings as input
2. **Multimodal Analysis**: Combines text transcription and prosody features using Google Health AI models
   - **MedASR**: Medical Automatic Speech Recognition for transcription
   - **MedGemma**: Medical language model for clinical text analysis and insights
3. **Multiple Classifiers**: 
   - PHQ-9 (Depression) scoring (0-27)
   - Anxiety risk assessment
   - PTSD risk assessment
4. **Medical Reports**: Automated report generation with MedGemma-enhanced clinical summaries
5. **Alert System**: Automatic alerts for high-risk cases with MedGemma-enhanced suicide risk detection
6. **Database Storage**: Persistent storage of all calls, scores, reports, and alerts
7. **Hybrid Model Strategy**: Uses MedGemma 4B for routine cases, automatically upgrades to 27B for critical cases

## Architecture

```
main_app.py          # Main application entry point
├── multimodal_analyzer.py    # Text + prosody analysis
├── classifiers.py            # PHQ-9, Anxiety, PTSD classifiers
├── audio_processor.py        # Audio extraction and prosody features
├── database.py               # SQLite database for storage
├── alert_system.py           # Risk assessment and alert generation
├── report_generator.py       # Medical report generation
└── config.py                # Configuration settings
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure ffmpeg is installed on your system:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

## Usage

### Basic Usage

```python
from main_app import MentalHealthScreeningApp
from datetime import datetime

# Initialize application
app = MentalHealthScreeningApp()

# Process a phone call
result = app.process_call(
    call_path="path/to/call_recording.wav",
    soldier_id="SOLDIER_001",
    call_timestamp=datetime.now()
)

# Access results
print(f"PHQ-9 Score: {result['scores']['phq9_score']}")
print(f"Risk Level: {result['risk_assessment']['severity']}")
print(f"Report: {result['report_path']}")
```

### Training Classifiers

To train the classifiers on your dataset:

```python
from classifiers import PHQ9Classifier, AnxietyClassifier, PTSDClassifier
import pandas as pd

# Load training data
df = pd.read_csv("train.csv")

# Prepare data
texts = df["transcript"].tolist()
prosody_list = [extract_prosody_features(wav_path) for wav_path in df["wav_path"]]
phq9_labels = df["phq9_total"].values
anxiety_labels = (df["anxiety_score"] >= threshold).astype(int)
ptsd_labels = (df["ptsd_score"] >= threshold).astype(int)

# Train classifiers
phq9_clf = PHQ9Classifier()
phq9_clf.fit(texts, prosody_list, phq9_labels)
phq9_clf.save("artifacts/phq9_classifier.pkl")

anxiety_clf = AnxietyClassifier()
anxiety_clf.fit(texts, prosody_list, anxiety_labels)
anxiety_clf.save("artifacts/anxiety_classifier.pkl")

ptsd_clf = PTSDClassifier()
ptsd_clf.fit(texts, prosody_list, ptsd_labels)
ptsd_clf.save("artifacts/ptsd_classifier.pkl")
```

## Configuration

Edit `config.py` to customize:
- Risk thresholds
- Alert recipients
- Model parameters
- File paths

## Alert System

The system automatically generates alerts for:
- **Critical**: Suicide risk indicators detected
- **High**: Severe depression (PHQ-9 ≥ 15), severe anxiety (≥80%), severe PTSD (≥80%)
- **Moderate**: Moderate depression (PHQ-9 ≥ 10), moderate anxiety/PTSD (≥60%)

Alerts are stored in the database and can be sent to configured recipients.

## Database Schema

- **calls**: Call records with metadata
- **scores**: Screening scores for each call
- **reports**: Generated medical reports
- **alerts**: Alert records

## File Structure

```
MEDGEMMA/
├── main_app.py              # Main application
├── config.py                # Configuration
├── multimodal_analyzer.py   # Multimodal analysis
├── classifiers.py           # ML classifiers
├── audio_processor.py       # Audio processing
├── database.py              # Database handler
├── alert_system.py          # Alert system
├── report_generator.py      # Report generation
├── requirements.txt         # Dependencies
├── data/                    # Database and data files
├── artifacts/               # Trained models and cache
├── reports/                 # Generated reports
└── wav_cache/               # Audio cache
```

## Models Used

### MedASR (`google/medasr`)
- **Purpose**: Speech-to-text transcription of phone calls
- **Location**: `multimodal_analyzer.py`
- **Usage**: Converts audio to text for analysis

### MedGemma (`google/medgemma-2-4b-it` or `google/medgemma-2-27b-it`)
- **Purpose**: Medical text analysis, clinical reasoning, and report generation
- **Location**: `medgemma_analyzer.py`, integrated into `multimodal_analyzer.py`, `alert_system.py`, `report_generator.py`
- **Usage**: 
  - Analyzes transcripts for medical insights
  - Enhances suicide risk detection
  - Generates clinical summaries for reports
  - Automatically uses 27B for critical cases (configurable)

### Model Selection Strategy
- **Default**: MedGemma 4B (faster, lower resource requirements)
- **Critical Cases**: Automatically upgrades to MedGemma 27B for high-risk cases
- **Configuration**: See `config.py` for `MEDGEMMA_MODEL_ID` and `USE_MEDGEMMA_27B_FOR_CRITICAL`

See `MEDGEMMA_COMPARISON.md` for detailed comparison of 4B vs 27B models.

## Notes

- The application uses Google's MedASR model for medical speech recognition
- MedGemma provides enhanced medical text analysis and clinical insights
- Prosody features include pitch, speaking rate, pauses, and energy
- All results are stored in SQLite database for tracking and review
- Medical reports are generated with MedGemma-enhanced clinical summaries

## Security & Privacy

⚠️ **Important**: This application handles sensitive medical data. Ensure:
- Proper encryption of stored data
- Secure transmission of phone call recordings
- Compliance with HIPAA and military data protection regulations
- Access controls for database and reports
