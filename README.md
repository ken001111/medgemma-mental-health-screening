## Overview

This repo implements a **mental health screening system for soldiers** based on short phone interviews.

It:

- Records or ingests a soldier’s call (≈2 minutes)
- Uses **MedASR** for medical speech recognition
- Extracts **prosody features** (pitch, rate, pauses, energy)
- Runs **classifiers** for PHQ‑9, anxiety, and PTSD risk
- Uses **MedGemma** for medical reasoning and clinical summaries
- Stores everything in **SQLite**
- Exposes:
  - A **CLI / Python API**
  - A **REST API**
  - An **iPhone-style web GUI** with:
    - Record button
    - Daily recordings “database” view
    - “Send to clinician” with consent
    - Post-screening **chat** powered by Google models (Gemini API with local MedGemma fallback)

---

## How MedGemma is used

MedGemma is the medical language model at the heart of the analysis. It is loaded via Hugging Face in `medgemma_analyzer.py` using the model ID from `config.py`:

- **Config:** `MEDGEMMA_MODEL_ID = "google/medgemma-2-4b-it"` (4B) by default.
- **Loader:** `MedGemmaAnalyzer` uses `AutoTokenizer` + `AutoModelForCausalLM` and runs on CPU or GPU.

MedGemma is used in three main ways:

- **1. Transcript analysis (`analyze_transcript`)**
  - Input: call transcript + optional scores (PHQ‑9, anxiety, PTSD).
  - Output: dict with:
    - `medical_insights`: bullet-like clinical points.
    - `risk_indicators`: flags for concerning phrases or themes.
    - `clinical_summary`: 1–2 paragraph narrative of the call.
    - `medgemma_available`: whether the model actually ran.
  - This is invoked inside `MentalHealthScreeningApp.process_call()` via `MultimodalAnalyzer` and feeds into risk assessment and report generation.

- **2. Clinical summary for reports (`generate_clinical_summary`)**
  - Builds a prompt with transcript + scores and lets MedGemma write a concise clinical summary.
  - `report_generator.py` uses this to produce the text reports saved under `reports/` and indexed in the DB.
  - If MedGemma cannot load, it falls back to `_generate_basic_summary` (rule-based summary using scores only).

- **3. Suicide / high‑risk reasoning (`detect_suicide_risk`)**
  - Asks MedGemma to reason about suicide risk level, specific indicators, and reasoning.
  - `alert_system.py` uses this to create high/critical alerts stored in the `alerts` table.
  - If MedGemma is unavailable, it falls back to keyword search + conservative defaults.

- **4. Local chat fallback in the GUI**
  - The GUI chat endpoint (`/api/chat` in `gui_app.py`) first tries Google’s **Gemini API**.
  - If the Gemini API is unavailable (404/model not found, 429/quota, etc.), it falls back to `_chat_with_medgemma_local`, which:
    - Reuses the same MedGemma model already loaded locally.
    - Builds a short chat-style prompt with severity, PHQ‑9, report snippet, transcript snippet, and recent messages.
    - Returns a brief, supportive answer with explicit **“not a clinician”** disclaimers.

In all cases, MedGemma is used for **clinical reasoning and summarization** — scores come from your classifiers, not from MedGemma directly.

---

## Code architecture

### Core pipeline

```text
main_app.py               # MentalHealthScreeningApp (main orchestration)
├── multimodal_analyzer.py  # MedASR + prosody + MedGemma analysis
├── medgemma_analyzer.py    # MedGemma integration (analysis, summaries, suicide risk)
├── classifiers.py          # PHQ‑9, Anxiety, PTSD classifiers
├── audio_processor.py      # Extract audio, compute prosody features
├── database.py             # SQLite DB (calls, scores, reports, alerts, clinician sends)
├── alert_system.py         # Risk assessment and alert generation
├── report_generator.py     # Text report generation using MedGemma summaries
└── config.py               # Paths, thresholds, model IDs, Gemini config
```

### Interfaces

```text
run_test.py        # Command-line entry point for single recording
api_handler.py     # REST API (process_call, history, alerts)
gui_app.py         # Flask app for iPhone-style GUI (record, database screen, chat)
pipeline_demo.ipynb# Notebook demo of the same pipeline
record_sample.py   # Simple mic recorder to create a test MP3
train_classifiers.py # Optional: (re)train PHQ‑9 / anxiety / PTSD classifiers
test_minimal.py    # Quick environment / dependency sanity check
RUN.md             # Detailed run instructions and quick reference
```

I have removed only **generated report text files** in `reports/` (they are re‑created automatically). All Python modules listed above are in active use by the CLI/API/GUI.

---

## Installation

### 1. Prerequisites

- **Python**: 3.8+ (3.11 recommended).
- **ffmpeg**: required for audio processing.  
  - macOS: `brew install ffmpeg`  
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`

### 2. Virtual environment + dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Hugging Face auth (MedASR + MedGemma)

MedASR is gated; you must accept terms and log in:

1. Visit `https://huggingface.co/google/medasr` and click **“Agree and access repository”**.
2. Create a token at `https://huggingface.co/settings/tokens`.
3. In this repo (with venv activated):

```bash
hf auth login
```

If `hf` is not found:

```bash
.venv/bin/hf auth login
# or
python -m huggingface_hub.cli.hf auth login
```

### 4. Optional: configure thresholds and recipients

Edit `config.py`:

- **Models**: `MEDGEMMA_MODEL_ID`, `USE_MEDGEMMA_27B_FOR_CRITICAL`, `GEMINI_MODEL`.
- **Risk thresholds** for PHQ‑9, anxiety, PTSD.
- **Alert recipients** (`ALERT_EMAILS`, `ALERT_PHONE_NUMBERS`).
- Paths for `DATA_DIR`, `REPORTS_DIR`, `ARTIFACTS_DIR`, etc.

For more step‑by‑step instructions, see `RUN.md`.

---

## How to run

### 1. CLI – single recording

```bash
cd /path/to/MEDGEMMA
source .venv/bin/activate
python run_test.py path/to/your_recording.mp3
```

This:

- Runs the full pipeline (`MentalHealthScreeningApp.process_call`).
- Prints transcript, scores, and risk assessment.
- Writes a text report to `reports/`.
- Stores call, scores, and report info into `data/screening_database.db`.

### 2. Python usage

```python
from datetime import datetime
from main_app import MentalHealthScreeningApp

app = MentalHealthScreeningApp()

result = app.process_call(
    call_path="path/to/recording.mp3",
    soldier_id="SOLDIER_001",
    call_timestamp=datetime.now(),
)

print("Call ID:", result["call_id"])
print("Scores:", result.get("scores"))
print("Risk:", result.get("risk_assessment", {}).get("severity"))
print("Report path:", result.get("report_path"))
```

### 3. REST API

Start the API server:

```bash
source .venv/bin/activate
python api_handler.py
```

Default base URL: `http://0.0.0.0:5000`

Key endpoints:

- `GET /health` – health check.
- `POST /process_call` – process an uploaded audio file:
  - `file`: audio/video (MP3/WAV/MP4…)
  - `soldier_id`: ID for the soldier
  - optional `call_timestamp`: ISO timestamp.
- `GET /soldier_history/<soldier_id>` – past calls and results.
- `GET /alerts` – pending high‑risk alerts.

Example:

```bash
curl -X POST \
  -F "file=@/path/to/recording.mp3" \
  -F "soldier_id=SOLDIER_001" \
  http://localhost:5000/process_call
```

### 4. iPhone‑style GUI (recordings + database + chat)

This is the frontend you can use for demo videos.

Start it:

```bash
source .venv/bin/activate
python gui_app.py
```

Then open: `http://127.0.0.1:5001`  
Use a narrow window or DevTools device toolbar (e.g. iPhone 14) for the iPhone look.

#### Optional: enable Gemini chat

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# Optional override; default is models/gemini-flash-latest from config.py
export GEMINI_MODEL="models/gemini-flash-latest"

source .venv/bin/activate
python gui_app.py
```

If the Gemini API is unavailable (no quota / 404 model IDs), the chat endpoint will automatically fall back to **local MedGemma** where possible.

#### GUI flow

- **Screen 1 – Soldier Mental Health Screening**
  - Big **Record** button.
  - Tap to start/stop recording.
  - Audio is uploaded to `/api/process_recording` and run through the same pipeline as the CLI/API.

- **Result screen**
  - Shows **severity** (low / moderate / high).
  - Buttons:
    - **Chat about your results** – opens the chat screen for this call.
    - **View history** – opens the recordings “database” view.

- **Chat screen**
  - Subtitle: “Powered by Google AI • For soldiers”.
  - Sends your questions + call context to the backend:
    - Tries Gemini via `_chat_with_gemini`.
    - Falls back to `_chat_with_medgemma_local` if Gemini is unavailable.
  - “Back to database” returns to the recordings list.

- **Recordings screen (“database view”)**
  - Shows recordings **grouped by day**, with:
    - Time, severity chip, PHQ‑9 score.
    - **Chat** button per row.
    - **Send to clinician** button per row.
  - Consent popup is shown before marking a call as “sent to clinician” in the DB.

---

## Data & database

SQLite DB (default: `data/screening_database.db`) with tables:

- `calls`: basic call metadata, audio path, transcript.
- `scores`: PHQ‑9, anxiety, PTSD scores per call.
- `reports`: MedGemma‑based report text + file path.
- `alerts`: high/critical alerts with severity and message.
- `clinician_sends`: which calls were sent to clinician from the GUI.

The GUI uses `ScreeningDatabase.get_recordings_for_gui()` to fetch joined calls + scores + computed severity and “sent to clinician” flags, grouped by date.

---

## Training classifiers (optional)

Use `train_classifiers.py` if you have your own labeled dataset and want to re‑train PHQ‑9 / anxiety / PTSD models.

- Saves trained models under `artifacts/`.
- `MentalHealthScreeningApp` will load them automatically if present.

This is optional for the demo; pre‑trained models can be used as is.

---

## Security & privacy

This is a **research/demo** system for soldier mental health screening. For any real deployment:

- Encrypt the DB, audio files, and reports.
- Use HTTPS/TLS for all network endpoints.
- Restrict access to clinicians / authorized staff only.
- Align with **HIPAA** and relevant **military** data protection rules.

MedGemma and MedASR run locally via Hugging Face; ensure:

- Model weights and tokens are stored securely.
- Logs avoid raw PII or full transcripts wherever possible.
