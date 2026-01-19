#!/usr/bin/env python3
"""Complete test script for mental health screening"""
import os
import sys
from datetime import datetime

# ---- Environment sanity check (common macOS/zsh issue) ----
# Some shells alias `python` to a system/Homebrew python, which overrides venv activation.
# If that happens, imports like torch will fail even though they are installed in .venv.
def _ensure_running_inside_venv() -> None:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    in_venv = base_prefix != sys.prefix
    if in_venv:
        return

    print("✗ Not running inside the virtual environment.")
    print(f"  sys.executable: {sys.executable}")
    print("")
    print("Fix:")
    print('  source .venv/bin/activate')
    print('  python run_test.py /path/to/your_audio.wav')
    print("")
    print("If your shell aliases `python`, run explicitly:")
    print('  .venv/bin/python run_test.py /path/to/your_audio.wav')
    sys.exit(1)


_ensure_running_inside_venv()

# Import after venv check, so missing deps show up correctly
from main_app import MentalHealthScreeningApp

def main():
    # Check for test audio file
    if len(sys.argv) < 2:
        print("="*60)
        print("MENTAL HEALTH SCREENING TEST")
        print("="*60)
        print("\nUsage: python run_test.py <audio_file_path>")
        print("\nExample:")
        print("  python run_test.py test_audio.wav")
        print("  python run_test.py test_audio.mp3")
        print("  python run_test.py test_audio.mp4")
        print("\nSupported formats: WAV, MP3, MP4, M4A, etc.")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    print("="*60)
    print("MENTAL HEALTH SCREENING TEST")
    print("="*60)
    print(f"Audio file: {audio_file}")
    print(f"File size: {os.path.getsize(audio_file) / (1024*1024):.2f} MB")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    try:
        # Initialize app
        print("\n[1/2] Initializing application...")
        print("  (This may take a few minutes on first run - downloading models)")
        app = MentalHealthScreeningApp(load_classifiers=True)
        
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
        print(f"Soldier ID: {result['soldier_id']}")
        
        print(f"\n📝 Transcript Preview:")
        print("-" * 60)
        transcript = result['transcript']
        if len(transcript) > 300:
            print(transcript[:300] + "...")
            print(f"\n[Full transcript length: {len(transcript)} characters]")
        else:
            print(transcript)
        print("-" * 60)
        
        print(f"\n📊 Scores:")
        scores = result.get('scores', {})
        if scores:
            for key, value in scores.items():
                if isinstance(value, float):
                    if 'risk' in key.lower():
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  (No classifiers loaded - scores not available)")
            print("  Train classifiers using: python train_classifiers.py")
        
        print(f"\n⚠️  Risk Assessment:")
        risk_assessment = result.get('risk_assessment', {})
        print(f"  Severity: {risk_assessment.get('severity', 'unknown').upper()}")
        
        if risk_assessment.get('risk_factors'):
            print(f"  Risk Factors:")
            for factor in risk_assessment['risk_factors']:
                print(f"    - {factor}")
        
        alerts = result.get('alerts', [])
        print(f"\n🚨 Alerts Generated: {len(alerts)}")
        if alerts:
            for alert in alerts:
                print(f"  [{alert.get('severity', 'unknown').upper()}] {alert.get('message', '')}")
        
        print(f"\n📄 Report:")
        print(f"  Path: {result.get('report_path', 'N/A')}")
        if result.get('report_path') and os.path.exists(result['report_path']):
            print(f"  Size: {os.path.getsize(result['report_path']) / 1024:.2f} KB")
        
        print("\n💾 Database:")
        print(f"  Location: data/screening_database.db")
        if os.path.exists("data/screening_database.db"):
            print(f"  Size: {os.path.getsize('data/screening_database.db') / 1024:.2f} KB")
        
        print("="*60)
        print("\n✓ Test completed successfully!")
        print("\nTo view the full report:")
        if result.get('report_path'):
            print(f"  cat {result['report_path']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
