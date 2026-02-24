#!/usr/bin/env python3
"""
Record from your microphone and save as MP3 (or WAV).
Use this to create a sample for the mental health screening pipeline.

Usage:
  python record_sample.py                    # 2 min, saves as my_recording.mp3
  python record_sample.py --duration 60      # 1 min
  python record_sample.py --output call.mp3
  python record_sample.py --format wav       # save as WAV instead of MP3
"""
import argparse
import subprocess
import sys
import os
from datetime import datetime

# Default: 2 minutes for screening call
DEFAULT_DURATION = 120
DEFAULT_OUTPUT = "my_recording.mp3"


def get_ffmpeg_input():
    """Return ffmpeg input flag for default mic (macOS: avfoundation, Linux: alsa/pulse)."""
    if sys.platform == "darwin":
        return "-f", "avfoundation", "-i", ":0"
    if sys.platform.startswith("linux"):
        # Prefer pulse, then default
        return "-f", "pulse", "-i", "default"
    # Windows: dshow
    return "-f", "dshow", "-i", "audio=default"


def record_mp3(output_path: str, duration_seconds: int) -> bool:
    """Record from default microphone to MP3 using ffmpeg."""
    *input_fmt, = get_ffmpeg_input()
    cmd = [
        "ffmpeg", "-y",
        *input_fmt,
        "-t", str(duration_seconds),
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "libmp3lame",
        "-q:a", "4",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode() if e.stderr else e)
        return False
    except FileNotFoundError:
        print("ffmpeg not found. Install it (e.g. brew install ffmpeg).")
        return False


def record_wav(output_path: str, duration_seconds: int) -> bool:
    """Record from default microphone to WAV using ffmpeg."""
    *input_fmt, = get_ffmpeg_input()
    cmd = [
        "ffmpeg", "-y",
        *input_fmt,
        "-t", str(duration_seconds),
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode() if e.stderr else e)
        return False
    except FileNotFoundError:
        print("ffmpeg not found. Install it (e.g. brew install ffmpeg).")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Record from microphone to MP3 or WAV for the screening pipeline."
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Recording length in seconds (default: {DEFAULT_DURATION}).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=("mp3", "wav"),
        default="mp3",
        help="Output format: mp3 or wav (default: mp3).",
    )
    args = parser.parse_args()

    if args.duration <= 0 or args.duration > 600:
        print("Duration must be between 1 and 600 seconds.")
        sys.exit(1)

    # If output has no extension, add one
    out = args.output
    if not out.lower().endswith((".mp3", ".wav")):
        out = f"{out}.{args.format}"

    print(f"Recording for {args.duration} seconds. Speak into your microphone.")
    print(f"Output: {os.path.abspath(out)}")
    print("Press Ctrl+C to stop early.\n")

    try:
        if args.format == "mp3":
            ok = record_mp3(out, args.duration)
        else:
            ok = record_wav(out, args.duration)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            size = os.path.getsize(out) / (1024 * 1024)
            print(f"Partial recording saved to {out} ({size:.2f} MB)")
            print(f"Run the pipeline: python run_test.py {out}")
        else:
            print("No file was saved. Run again and let the recording finish, or stop after a few seconds to save a short clip.")
        sys.exit(0)

    if ok:
        size = os.path.getsize(out) / (1024 * 1024)
        print(f"\nDone. Saved to {out} ({size:.2f} MB)")
        print(f"Run the pipeline: python run_test.py {out}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
