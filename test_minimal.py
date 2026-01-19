#!/usr/bin/env python3
"""Minimal test to verify installation"""
import os
import sys

print("Testing installation...")
print("=" * 60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"sys.prefix: {sys.prefix}")
print(f"base_prefix: {getattr(sys, 'base_prefix', '')}")
in_venv = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
print(f"In virtualenv: {in_venv}")
print("-" * 60)

# Test imports
try:
    import torch
    print("✓ PyTorch installed")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("✗ PyTorch import failed")
    print(f"  Error: {type(e).__name__}: {e}")
    print("")
    print("Common fix on zsh: your `python` may be aliased to a system/Homebrew Python.")
    print("Run the venv interpreter explicitly:")
    print('  .venv/bin/python -c "import torch; print(torch.__version__)"')
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

try:
    import sklearn
    print("✓ Scikit-learn installed")
except ImportError:
    print("✗ Scikit-learn not installed")
    sys.exit(1)

try:
    import pandas
    print("✓ Pandas installed")
except ImportError:
    print("✗ Pandas not installed")
    sys.exit(1)

try:
    import soundfile
    print("✓ Soundfile installed")
except ImportError:
    print("✗ Soundfile not installed")
    sys.exit(1)

# Test ffmpeg
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          text=True,
                          timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✓ ffmpeg installed")
        print(f"  {version_line}")
    else:
        print("✗ ffmpeg not found")
        sys.exit(1)
except FileNotFoundError:
    print("✗ ffmpeg not installed")
    print("  Install with: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Linux)")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print("✗ ffmpeg check timed out")
    sys.exit(1)

# Test directory structure
print("\nChecking directory structure...")
required_dirs = ['data', 'artifacts', 'reports', 'wav_cache']
for dir_name in required_dirs:
    os.makedirs(dir_name, exist_ok=True)
    if os.path.exists(dir_name):
        print(f"✓ Directory '{dir_name}' ready")
    else:
        print(f"✗ Directory '{dir_name}' creation failed")

# Test config import
try:
    from config import MEDASR_ID, MEDGEMMA_MODEL_ID
    print(f"\n✓ Configuration loaded")
    print(f"  MedASR model: {MEDASR_ID}")
    print(f"  MedGemma model: {MEDGEMMA_MODEL_ID}")
except Exception as e:
    print(f"\n✗ Configuration error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All basic requirements met!")
print("="*60)
print("\nNext steps:")
print("1. Prepare a test audio file (WAV, MP3, or MP4)")
print("2. Run: python run_test.py your_audio_file.wav")
print("3. Or use the main app directly (see TESTING_GUIDE.md)")
print("\nNote: First run will download models (~8-9GB total)")
print("This may take 10-30 minutes depending on your internet speed.")
