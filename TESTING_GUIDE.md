# Testing Guide: How to Test the Application

## 📋 Prerequisites

### 1. **System Requirements**

- **Python**: 3.8 or higher
- **GPU** (recommended): CUDA-capable GPU with at least 8GB VRAM for MedGemma 4B
  - For CPU-only: Will work but much slower
- **RAM**: At least 16GB recommended
- **Storage**: ~10GB free space (for models and cache)
- **OS**: macOS, Linux, or Windows (with WSL recommended)

### 2. **Required Software**

- **ffmpeg**: For audio processing
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # Windows (using Chocolatey)
  choco install ffmpeg
  ```

### 3. **Python Dependencies**

Install all Python packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install torch>=2.0.0
pip install ffmpeg-python>=0.2.0
pip install soundfile>=0.12.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install librosa>=0.10.0
pip install flask>=2.3.0
pip install werkzeug>=2.3.0
```

---

## 🔧 Setup Steps

### Step 1: Clone/Download Repository

```bash
git clone https://github.com/ken001111/medgemma-mental-health-screening.git
cd medgemma-mental-health-screening
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2.5 (macOS/zsh): Ensure you're using the venv Python

If your shell has a `python` alias, it can override the venv and cause errors like `No module named 'torch'`.

Run:

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
```

If `sys.executable` is **not** inside your venv folder, run Python explicitly:

```bash
.venv/bin/python test_minimal.py
.venv/bin/python run_test.py your_audio_file.wav
```

### Step 3: Verify ffmpeg Installation

```bash
ffmpeg -version
```

Should show ffmpeg version information.

### Step 4: Configure Settings (Optional)

Edit `config.py` if needed:
- Alert email addresses
- Database path
- Model paths

---

## 🎯 Testing Options

### Option 1: Quick Test (Without Trained Classifiers)

This tests the basic pipeline without ML classifiers.

**What you need:**
- A phone call audio file (WAV, MP3, MP4, etc.)
- Duration: ~2 minutes recommended

**Test script:**

Create `test_basic.py`:
```python
from main_app import MentalHealthScreeningApp
from datetime import datetime

# Initialize app (will load models)
print("Initializing application...")
app = MentalHealthScreeningApp(load_classifiers=False)  # Skip classifier loading

# Test with your audio file
call_path = "path/to/your/call.wav"  # Replace with your file

try:
    result = app.process_call(
        call_path=call_path,
        soldier_id="TEST_SOLDIER_001",
        call_timestamp=datetime.now()
    )
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Call ID: {result['call_id']}")
    print(f"\nTranscript (first 200 chars):")
    print(result['transcript'][:200] + "...")
    
    if result.get('medgemma_analysis'):
        print(f"\nMedGemma Analysis Available: Yes")
        if result['medgemma_analysis'].get('medical_insights'):
            print(f"Medical Insights: {len(result['medgemma_analysis']['medical_insights'])}")
    
    print(f"\nRisk Assessment:")
    print(f"  Severity: {result['risk_assessment']['severity']}")
    print(f"  Risk Factors: {result['risk_assessment'].get('risk_factors', [])}")
    
    print(f"\nReport Path: {result['report_path']}")
    print("="*60)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
```

**Run:**
```bash
python test_basic.py
```

---

### Option 2: Full Test (With Trained Classifiers)

This requires trained classifiers. You have two options:

#### A. Train Your Own Classifiers

**What you need:**
- Training data CSV with columns:
  - `id`: Unique identifier
  - `video_path` or `audio_path`: Path to audio/video file
  - `phq9_total`: PHQ-9 score (0-27)
  - `anxiety_label`: Binary label (0/1) or score
  - `ptsd_label`: Binary label (0/1) or score

**Train classifiers:**
```bash
python train_classifiers.py \
    --data train.csv \
    --audio_dir /path/to/audio/files \
    --train_all
```

This will:
- Extract features from all audio files
- Train PHQ-9, Anxiety, and PTSD classifiers
- Save models to `artifacts/` directory

#### B. Use Pre-trained Classifiers (if available)

Place pre-trained models in `artifacts/`:
- `phq9_classifier.pkl`
- `anxiety_classifier.pkl`
- `ptsd_classifier.pkl`

Then run:
```python
from main_app import MentalHealthScreeningApp

app = MentalHealthScreeningApp(load_classifiers=True)  # Load classifiers
result = app.process_call("call.wav", "SOLDIER_001")
```

---

### Option 3: Test Individual Components

#### Test Audio Processing

Create `test_audio.py`:
```python
from audio_processor import extract_audio_from_call, extract_prosody_features

# Test audio extraction
wav_path = extract_audio_from_call("input_call.mp3")
print(f"Extracted audio to: {wav_path}")

# Test prosody extraction
prosody = extract_prosody_features(wav_path)
print("\nProsody Features:")
for key, value in prosody.items():
    print(f"  {key}: {value:.2f}")
```

#### Test Transcription

Create `test_transcription.py`:
```python
from multimodal_analyzer import MultimodalAnalyzer

analyzer = MultimodalAnalyzer(use_medgemma=False)  # Skip MedGemma for faster test

transcript = analyzer.transcribe("test_call.wav")
print(f"Transcript: {transcript}")
```

#### Test MedGemma

Create `test_medgemma.py`:
```python
from medgemma_analyzer import create_medgemma_analyzer

medgemma = create_medgemma_analyzer(use_27b=False)

test_transcript = """
I've been feeling really down lately. I can't sleep, 
and I've lost interest in everything. Sometimes I 
wonder if it's even worth it.
"""

analysis = medgemma.analyze_transcript(test_transcript)
print("Medical Insights:", analysis.get('medical_insights', []))
print("Risk Indicators:", analysis.get('risk_indicators', []))
```

---

## 📁 Test Data Setup

### Create Test Directory Structure

```bash
mkdir -p test_data
mkdir -p test_data/audio
mkdir -p test_data/results
```

### Sample Test Audio

You can use:
1. **Your own recordings**: Record a 2-minute test call
2. **Public datasets**: 
   - DAIC-WOZ (Depression dataset)
   - Audio samples from research papers
3. **Synthetic test**: Create a simple WAV file for basic testing

### Minimal Test File

Create `test_minimal.py`:
```python
"""Minimal test to verify installation"""
import os
import sys

print("Testing installation...")

# Test imports
try:
    import torch
    print("✓ PyTorch installed")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    print("✓ Transformers installed")
except ImportError:
    print("✗ Transformers not installed")
    sys.exit(1)

try:
    import librosa
    print("✓ Librosa installed")
except ImportError:
    print("✗ Librosa not installed")
    sys.exit(1)

# Test ffmpeg
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          text=True)
    if result.returncode == 0:
        print("✓ ffmpeg installed")
    else:
        print("✗ ffmpeg not found")
except FileNotFoundError:
    print("✗ ffmpeg not installed")
    sys.exit(1)

# Test directory structure
required_dirs = ['data', 'artifacts', 'reports', 'wav_cache']
for dir_name in required_dirs:
    os.makedirs(dir_name, exist_ok=True)
    print(f"✓ Directory '{dir_name}' ready")

print("\n✓ All basic requirements met!")
print("\nNext steps:")
print("1. Download models (will happen automatically on first run)")
print("2. Prepare test audio file")
print("3. Run test_basic.py")
```

Run:
```bash
python test_minimal.py
```

---

## 🚀 Quick Start Testing

### Step-by-Step Quick Test

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare test audio:**
   - Get a 2-minute audio file (WAV, MP3, or MP4)
   - Place it in the project directory

3. **Run basic test:**
   ```python
   from main_app import MentalHealthScreeningApp
   
   app = MentalHealthScreeningApp(load_classifiers=False)
   result = app.process_call("your_test_audio.wav", "TEST_001")
   print(result)
   ```

4. **Check results:**
   - Transcript: `result['transcript']`
   - Report: `result['report_path']`
   - Database: Check `data/screening_database.db`

---

## 🔍 What to Expect

### First Run

**Model Downloads:**
- MedASR model (~500MB): Downloads automatically from HuggingFace
- MedGemma 4B model (~8GB): Downloads automatically from HuggingFace
- **Note**: First run will take time to download models

**Expected Output:**
```
Initializing Mental Health Screening Application...
Loading MedASR model on cuda...
MedASR model loaded successfully.
Loading MedGemma model (google/medgemma-2-4b-it) on cuda...
MedGemma model loaded successfully.

Processing call abc-123 for soldier TEST_001...
Step 1: Extracting audio...
Step 2: Performing multimodal analysis...
Step 2.5: Enhanced MedGemma analysis with scores...
Step 3: Running mental health classifiers...
Step 4: Assessing risk and generating alerts...
Step 5: Saving to database...
Step 6: Generating medical report...

✓ Call processing complete!
  Report saved to: reports/report_abc123_20260119.txt
```

### Common Issues

1. **CUDA Out of Memory:**
   - Use CPU: Set `device="cpu"` in `MultimodalAnalyzer()`
   - Or use smaller batch sizes

2. **Model Download Fails:**
   - Check internet connection
   - Verify HuggingFace access
   - May need to login: `huggingface-cli login`

3. **ffmpeg Not Found:**
   - Install ffmpeg (see Prerequisites)
   - Verify with `ffmpeg -version`

4. **No Classifiers:**
   - Expected if you haven't trained them
   - Set `load_classifiers=False` in `MentalHealthScreeningApp()`
   - Or train using `train_classifiers.py`

---

## 📊 Testing Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] ffmpeg installed and working
- [ ] Test audio file prepared
- [ ] Models download successfully (first run)
- [ ] Audio extraction works
- [ ] Transcription works (MedASR)
- [ ] Prosody extraction works
- [ ] MedGemma analysis works (optional)
- [ ] Database created successfully
- [ ] Report generated successfully
- [ ] Can query database

---

## 🎬 Example Test Script

Create `run_test.py`:
```python
#!/usr/bin/env python3
"""Complete test script"""
import os
import sys
from datetime import datetime
from main_app import MentalHealthScreeningApp

def main():
    # Check for test audio file
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <audio_file_path>")
        print("\nExample:")
        print("  python run_test.py test_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print("="*60)
    print("MENTAL HEALTH SCREENING TEST")
    print("="*60)
    print(f"Audio file: {audio_file}")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    try:
        # Initialize app
        print("\n[1/2] Initializing application...")
        app = MentalHealthScreeningApp(load_classifiers=False)
        
        # Process call
        print("\n[2/2] Processing call...")
        result = app.process_call(
            call_path=audio_file,
            soldier_id="TEST_SOLDIER_001",
            call_timestamp=datetime.now()
        )
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Call ID: {result['call_id']}")
        print(f"\nTranscript Preview:")
        print("-" * 60)
        print(result['transcript'][:300] + "..." if len(result['transcript']) > 300 else result['transcript'])
        print("-" * 60)
        
        print(f"\nScores:")
        for key, value in result['scores'].items():
            print(f"  {key}: {value}")
        
        print(f"\nRisk Assessment:")
        print(f"  Severity: {result['risk_assessment']['severity'].upper()}")
        if result['risk_assessment'].get('risk_factors'):
            print(f"  Risk Factors:")
            for factor in result['risk_assessment']['risk_factors']:
                print(f"    - {factor}")
        
        print(f"\nAlerts Generated: {len(result['alerts'])}")
        if result['alerts']:
            for alert in result['alerts']:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        
        print(f"\nReport: {result['report_path']}")
        print("="*60)
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python run_test.py your_audio_file.wav
```

---

## 📝 Next Steps After Testing

1. **Train Classifiers**: Use `train_classifiers.py` with your labeled data
2. **Configure Alerts**: Update `config.py` with real email addresses
3. **Set up API**: Use `api_handler.py` for REST API access
4. **Deploy**: Set up for production use

---

## 🆘 Troubleshooting

### Models Won't Download
```bash
# Login to HuggingFace (if needed)
huggingface-cli login

# Or set token
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### GPU Issues
```python
# Force CPU usage
analyzer = MultimodalAnalyzer(device="cpu")
```

### Database Locked
```bash
# Close any connections, then retry
# Or delete database to recreate
rm data/screening_database.db
```

---

## 📚 Additional Resources

- **README.md**: General project overview
- **HOW_IT_WORKS.md**: Detailed code explanation
- **MEDGEMMA_COMPARISON.md**: Model comparison guide
- **train_classifiers.py**: Training script documentation

Good luck with testing! 🚀
