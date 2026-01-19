# How the Code Works: Complete System Overview

## 🏗️ System Architecture

The application follows a **pipeline architecture** where a phone call flows through multiple stages:

```
Phone Call → Audio Extraction → Multimodal Analysis → Classification → Risk Assessment → Alerting → Report Generation → Database Storage
```

---

## 📋 Step-by-Step Processing Flow

### **Step 1: Application Initialization** (`main_app.py`)

When you create a `MentalHealthScreeningApp` instance:

```python
app = MentalHealthScreeningApp()
```

**What happens:**
1. **Loads MedASR model** (`google/medasr`) for speech-to-text transcription
2. **Loads MedGemma model** (`google/medgemma-2-4b-it`) for medical text analysis
3. **Initializes database** (SQLite) for storing results
4. **Creates alert system** for risk monitoring
5. **Sets up report generator** for medical reports
6. **Loads pre-trained classifiers** (PHQ-9, Anxiety, PTSD) if available

**Key Components:**
- `MultimodalAnalyzer`: Handles audio → text + prosody extraction
- `PHQ9Classifier`, `AnxietyClassifier`, `PTSDClassifier`: ML models for scoring
- `AlertSystem`: Monitors for high-risk cases
- `ReportGenerator`: Creates clinician reports
- `ScreeningDatabase`: Stores all data

---

### **Step 2: Process a Phone Call** (`main_app.py::process_call()`)

When you call:
```python
result = app.process_call(call_path="call.wav", soldier_id="SOLDIER_001")
```

#### **2.1 Audio Extraction** (`audio_processor.py`)

```python
wav_path = extract_audio_from_call(call_path)
```

**What happens:**
- Uses **ffmpeg** to convert phone call (any format) → WAV
- Normalizes to 16kHz mono audio
- Saves to `wav_cache/` directory
- Returns path to WAV file

**Why:** Standardizes audio format for processing

---

#### **2.2 Multimodal Analysis** (`multimodal_analyzer.py`)

```python
analysis = self.analyzer.analyze_call(wav_path)
```

**Two parallel processes:**

**A. Text Transcription (MedASR)**
```python
transcript = self.transcribe(wav_path)
```
- Loads audio into MedASR model
- Generates text transcript (max 768 tokens for ~2 min call)
- Returns: `"I've been feeling really down lately..."`

**B. Prosody Feature Extraction** (`audio_processor.py`)
```python
prosody_features = extract_prosody_features(wav_path)
```
- Uses **librosa** to extract acoustic features:
  - **Pitch** (fundamental frequency): Voice tone
  - **Speaking rate**: Words per minute
  - **Energy**: Voice volume/intensity
  - **Pauses**: Silence ratio
  - **Jitter**: Voice stability
  - **Spectral centroid**: Voice quality
- Returns dictionary: `{"mean_pitch": 180.5, "speaking_rate": 120, ...}`

**Why multimodal?** 
- **Text** captures what they say (content)
- **Prosody** captures how they say it (emotion, stress indicators)

---

#### **2.3 MedGemma Medical Analysis** (`medgemma_analyzer.py`)

```python
medgemma_analysis = self.analyzer.medgemma.analyze_transcript(transcript, scores)
```

**What happens:**
- Sends transcript + scores to MedGemma model
- MedGemma analyzes for:
  - Medical insights (symptoms, indicators)
  - Risk factors
  - Clinical reasoning
- Returns structured analysis

**Example output:**
```python
{
    "medical_insights": [
        "Patient reports persistent low mood",
        "Sleep disturbances mentioned"
    ],
    "risk_indicators": ["depression", "hopeless"],
    "clinical_summary": "Patient presents with symptoms consistent with..."
}
```

---

#### **2.4 Classification** (`classifiers.py`)

Three separate classifiers run:

**A. PHQ-9 Classifier** (Depression)
```python
phq9_score = self.phq9_classifier.predict_phq9_score([transcript], [prosody_features])
```

**How it works:**
1. **Text Features**: TF-IDF vectorization (20,000 features, 1-2 word n-grams)
   - Converts text → numerical features
   - Example: "feeling sad" → [0.0, 0.0, 0.5, 0.0, ...]
2. **Prosody Features**: Normalized acoustic features (9 features)
   - Example: [0.8, -0.2, 1.1, ...]
3. **Feature Combination**: Concatenates text + prosody features
   - Combined vector: [text_features (20K) + prosody_features (9)]
4. **Prediction**: Logistic Regression with calibration
   - Outputs: PHQ-9 score (0-27)

**B. Anxiety Classifier**
```python
anxiety_risk = self.anxiety_classifier.predict_anxiety_risk([transcript], [prosody_features])
```
- Same process, outputs probability (0-1)

**C. PTSD Classifier**
```python
ptsd_risk = self.ptsd_classifier.predict_ptsd_risk([transcript], [prosody_features])
```
- Same process, outputs probability (0-1)

**Why TF-IDF + Logistic Regression?**
- **TF-IDF**: Captures important words/phrases in medical context
- **Logistic Regression**: Fast, interpretable, works well with high-dimensional features
- **Calibration**: Ensures probabilities are well-calibrated for medical thresholds

---

#### **2.5 Risk Assessment** (`alert_system.py`)

```python
risk_assessment = self.alert_system.assess_risk_level(
    phq9_score=scores.get("phq9_score"),
    anxiety_risk=scores.get("anxiety_risk"),
    ptsd_risk=scores.get("ptsd_risk"),
    transcript=transcript
)
```

**What happens:**

1. **Suicide Risk Detection**:
   - Keyword matching: "suicide", "kill myself", etc.
   - **Enhanced with MedGemma**: Clinical reasoning beyond keywords
   - Returns: `(is_risk: bool, message: str)`

2. **Severity Assessment**:
   - **Critical**: Suicide risk detected
   - **High**: PHQ-9 ≥ 15, Anxiety ≥ 80%, PTSD ≥ 80%
   - **Moderate**: PHQ-9 ≥ 10, Anxiety ≥ 60%, PTSD ≥ 60%
   - **Low**: Below thresholds

3. **Risk Factors Compilation**:
   - Lists all identified risk factors
   - Example: `["Severe depression (PHQ-9: 18.5)", "Moderate anxiety (risk: 0.65)"]`

**Output:**
```python
{
    "severity": "high",
    "alerts": [...],
    "risk_factors": [...],
    "requires_immediate_attention": True
}
```

---

#### **2.6 Alert Generation** (`alert_system.py`)

```python
if risk_assessment["requires_immediate_attention"]:
    alerts = self.alert_system.generate_alert(call_id, risk_assessment)
```

**What happens:**
- Creates alert records in database
- Determines recipients based on severity:
  - **Critical**: Emails + Phone numbers
  - **High**: Emails only
- Stores alert for tracking

**Alert types:**
- `suicide_risk`: Critical alerts
- `depression`: High severity depression
- `anxiety`: High severity anxiety
- `ptsd`: High severity PTSD

---

#### **2.7 Database Storage** (`database.py`)

```python
self.database.add_call(...)
self.database.add_scores(...)
self.database.add_report(...)
```

**Four tables:**

1. **`calls`**: Call metadata
   - call_id, soldier_id, timestamp, duration, audio_path, transcript

2. **`scores`**: Screening scores
   - call_id, phq9_score, anxiety_risk, ptsd_risk

3. **`reports`**: Medical reports
   - call_id, report_content, report_path

4. **`alerts`**: Alert records
   - call_id, alert_type, severity, message, sent_to, acknowledged

**Why SQLite?**
- Lightweight, no server needed
- Perfect for single-deployment applications
- Easy to query and backup

---

#### **2.8 Report Generation** (`report_generator.py`)

```python
report = self.report_generator.generate_report(...)
```

**What happens:**

1. **Basic Report Sections**:
   - Call information (ID, soldier, timestamp)
   - Screening scores with severity labels
   - Risk assessment summary
   - Full transcript

2. **MedGemma Enhancement** (if available):
   - Clinical summary generated by MedGemma
   - Medical insights extracted
   - Risk indicators identified
   - Uses **27B model** for high-risk cases (if configured)

3. **File Output**:
   - Saves to `reports/report_{call_id}_{timestamp}.txt`
   - Stored in database

**Example Report:**
```
MENTAL HEALTH SCREENING REPORT
==================================================

CALL INFORMATION:
Call ID: abc-123-def
Soldier ID: SOLDIER_001
Call Date/Time: 2026-01-19 15:30:00

SCREENING SCORES:
PHQ-9 Score: 18.5/27 (Severe)
Anxiety Risk: 72% (Moderate)
PTSD Risk: 45% (Low)

RISK ASSESSMENT:
Overall Severity: HIGH
Risk Factors Identified:
  - Severe depression (PHQ-9: 18.5)
  - Moderate anxiety (risk: 0.72)

CLINICAL ANALYSIS (MedGemma):
Patient presents with symptoms consistent with major depressive disorder...
[MedGemma-generated clinical summary]

CALL TRANSCRIPT:
[Full transcript here]
```

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────┐
│ Phone Call  │
│  (audio)    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Extract Audio   │ ← ffmpeg converts to WAV
│  (audio_processor)│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Multimodal      │
│   Analysis      │
│                 │
│  ┌───────────┐  │
│  │ MedASR    │──┼──→ Transcript (text)
│  │ (Speech)  │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │ Prosody   │──┼──→ Features (pitch, rate, etc.)
│  │ Extraction│  │
│  └───────────┘  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ MedGemma        │ ← Medical text analysis
│   Analysis      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Classification  │
│                 │
│  ┌───────────┐  │
│  │ PHQ-9     │──┼──→ Depression Score (0-27)
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Anxiety   │──┼──→ Anxiety Risk (0-1)
│  └───────────┘  │
│  ┌───────────┐  │
│  │ PTSD      │──┼──→ PTSD Risk (0-1)
│  └───────────┘  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Risk Assessment │ ← Checks thresholds, suicide risk
│  (alert_system) │
└──────┬──────────┘
       │
       ├──→ High Risk? ──→ Generate Alerts
       │
       ▼
┌─────────────────┐
│ Report          │ ← MedGemma-enhanced report
│  Generation     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Database        │ ← Store everything
│  (SQLite)       │
└─────────────────┘
```

---

## 🧠 Key Algorithms & Techniques

### **1. TF-IDF Vectorization**
- **Term Frequency-Inverse Document Frequency**
- Converts text → numerical features
- Captures important words/phrases
- Example: "feeling hopeless" gets high weight if rare but meaningful

### **2. Feature Combination**
- **Text features** (20,000 dims) + **Prosody features** (9 dims)
- Normalized prosody features for scale matching
- Concatenated into single feature vector

### **3. Logistic Regression with Calibration**
- **Base model**: Logistic Regression (fast, interpretable)
- **Calibration**: Sigmoid calibration (3-fold CV)
- **Why calibrated?** Medical thresholds need accurate probabilities

### **4. Multimodal Fusion**
- Text (what they say) + Prosody (how they say it)
- Both contribute to final prediction
- Prosody can detect stress even when words don't indicate it

---

## 📊 Example: Processing a Call

**Input:**
- Phone call: `soldier_call_001.wav` (2 minutes)
- Soldier ID: `SOLDIER_001`

**Processing:**

1. **Audio extracted** → `wav_cache/soldier_001.wav`

2. **Transcription** (MedASR):
   ```
   "I've been having trouble sleeping. I feel really down all the time. 
   Nothing seems to help. Sometimes I think about... you know, ending it all."
   ```

3. **Prosody extracted**:
   ```python
   {
       "mean_pitch": 165.3,      # Lower than normal (depression indicator)
       "speaking_rate": 95,       # Slower than normal
       "silence_ratio": 0.15,     # More pauses
       "jitter": 0.08            # Voice instability
   }
   ```

4. **Classification**:
   - PHQ-9: **18.5/27** (Severe depression)
   - Anxiety: **0.72** (72% risk)
   - PTSD: **0.35** (35% risk)

5. **MedGemma Analysis**:
   - Medical insights: ["Sleep disturbance", "Persistent low mood", "Suicidal ideation"]
   - Risk indicators: ["depression", "suicide", "hopeless"]

6. **Risk Assessment**:
   - Severity: **CRITICAL** (suicide risk detected)
   - Alerts generated: 1 critical alert

7. **Report Generated**:
   - Saved to `reports/report_abc123_20260119.txt`
   - Includes MedGemma clinical summary

8. **Database Updated**:
   - Call record saved
   - Scores saved
   - Alert saved
   - Report saved

**Output:**
```python
{
    "call_id": "abc-123-def",
    "scores": {
        "phq9_score": 18.5,
        "anxiety_risk": 0.72,
        "ptsd_risk": 0.35
    },
    "risk_assessment": {
        "severity": "critical",
        "requires_immediate_attention": True
    },
    "alerts": [{
        "type": "suicide_risk",
        "severity": "critical",
        "message": "Suicide risk detected..."
    }],
    "report_path": "reports/report_abc123_20260119.txt"
}
```

---

## 🔧 Configuration (`config.py`)

Key settings you can adjust:

- **Models**: `MEDASR_ID`, `MEDGEMMA_MODEL_ID`
- **Thresholds**: `PHQ9_THRESHOLD_SEVERE`, `ANXIETY_THRESHOLD_SEVERE`
- **Alert recipients**: `ALERT_EMAILS`, `ALERT_PHONE_NUMBERS`
- **Paths**: Database, reports, cache directories

---

## 🚀 Usage Example

```python
from main_app import MentalHealthScreeningApp

# Initialize
app = MentalHealthScreeningApp()

# Process a call
result = app.process_call(
    call_path="phone_call.wav",
    soldier_id="SOLDIER_001"
)

# Check results
print(f"PHQ-9 Score: {result['scores']['phq9_score']}")
print(f"Risk Level: {result['risk_assessment']['severity']}")

# Get soldier history
history = app.get_soldier_history("SOLDIER_001", limit=5)

# Check pending alerts
alerts = app.get_pending_alerts()
```

---

## 🎯 Key Design Decisions

1. **Multimodal**: Text + Prosody for better accuracy
2. **Modular**: Each component is separate and testable
3. **Database-first**: All data persisted for tracking
4. **Alert-driven**: Automatic notifications for high-risk cases
5. **MedGemma integration**: Enhanced medical understanding
6. **Hybrid models**: 4B for speed, 27B for critical cases

This architecture ensures **accuracy**, **scalability**, and **safety** for mental health screening.
