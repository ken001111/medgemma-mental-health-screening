# Requirements:
#   pip install -U transformers accelerate torch ffmpeg-python soundfile scikit-learn pandas numpy
# Notes:
# - This script assumes English, ~2-minute self-answer videos.
# - It caches transcripts to avoid repeated ASR calls.

import os
import subprocess
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf

# ---- Config ----
MEDASR_ID = "google/medasr"  # MedASR model card: HAI-DEF + HF
SR = 16000
SEED = 42

# For ~2-minute clips, 256 tokens often truncates transcripts.
MAX_NEW_TOKENS = 768

# Cache file for transcripts (id -> transcript). This is written by the script.
TRANSCRIPT_CACHE_CSV = os.path.join("artifacts", "transcripts_cache.csv")

def extract_wav(video_path: str, wav_path: str, sr: int = SR):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@torch.no_grad()
def transcribe_medasr(wav_path: str, processor, model, device: str, max_new_tokens: int = MAX_NEW_TOKENS):
    audio, sr = sf.read(wav_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)

    # Deterministic decoding for reproducibility
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    return processor.batch_decode(out, skip_special_tokens=True)[0]

def main():
    df = pd.read_csv("train.csv")

    # Load transcript cache if present
    os.makedirs("artifacts", exist_ok=True)
    transcript_cache = {}
    if os.path.exists(TRANSCRIPT_CACHE_CSV):
        try:
            cache_df = pd.read_csv(TRANSCRIPT_CACHE_CSV)
            if "id" in cache_df.columns and "transcript" in cache_df.columns:
                transcript_cache = dict(zip(cache_df["id"].astype(str), cache_df["transcript"].fillna("")))
        except Exception:
            transcript_cache = {}

    df["label"] = (df["phq9_total"] >= 10).astype(int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MEDASR_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MEDASR_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    transcripts = []
    n_total = len(df)

    for i, row in df.iterrows():
        vid_id = str(row["id"])

        # Use cached transcript if available
        if vid_id in transcript_cache and isinstance(transcript_cache[vid_id], str) and transcript_cache[vid_id].strip():
            transcripts.append(transcript_cache[vid_id])
            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                print(f"[{i+1}/{n_total}] transcript: cache hit")
            continue

        vp = row["video_path"]
        wav = os.path.join("wav_cache", f"{vid_id}.wav")
        if not os.path.exists(wav):
            extract_wav(vp, wav)

        text = transcribe_medasr(wav, processor, model, device, max_new_tokens=MAX_NEW_TOKENS)
        transcripts.append(text)

        # Update cache incrementally (safer for long runs)
        transcript_cache[vid_id] = text
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            pd.DataFrame({"id": list(transcript_cache.keys()), "transcript": list(transcript_cache.values())}).to_csv(
                TRANSCRIPT_CACHE_CSV, index=False
            )
            print(f"[{i+1}/{n_total}] transcript: updated cache")

    df["transcript"] = transcripts

    Xtr, Xva, ytr, yva = train_test_split(
        df["transcript"], df["label"].values,
        test_size=0.2, stratify=df["label"].values, random_state=SEED
    )

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2
    )
    Xtr_t = tfidf.fit_transform(Xtr)
    Xva_t = tfidf.transform(Xva)

    base = LogisticRegression(max_iter=3000, n_jobs=-1)
    # Probability calibration is important for screening thresholds.
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr_t, ytr)

    p = clf.predict_proba(Xva_t)[:, 1]
    print("AUROC:", roc_auc_score(yva, p))
    print("AUPRC:", average_precision_score(yva, p))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    df[["id", "label", "phq9_total", "transcript"]].to_csv(os.path.join("artifacts", "transcripts.csv"), index=False)

    # Ensure final cache write
    pd.DataFrame({"id": list(transcript_cache.keys()), "transcript": list(transcript_cache.values())}).to_csv(
        TRANSCRIPT_CACHE_CSV, index=False
    )

if __name__ == "__main__":
    main()