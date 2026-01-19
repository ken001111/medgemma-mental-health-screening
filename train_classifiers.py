"""
Training script for mental health classifiers.
Trains PHQ-9, Anxiety, and PTSD classifiers on labeled data.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from config import SEED, ARTIFACTS_DIR
from classifiers import PHQ9Classifier, AnxietyClassifier, PTSDClassifier
from multimodal_analyzer import MultimodalAnalyzer
from audio_processor import extract_audio_from_call, extract_prosody_features


def load_training_data(csv_path: str, audio_dir: str = None):
    """
    Load training data from CSV file.
    
    Expected CSV columns:
    - id: Unique identifier
    - video_path or audio_path: Path to audio/video file
    - phq9_total: PHQ-9 score (0-27)
    - anxiety_label: Binary anxiety label (0/1) or score
    - ptsd_label: Binary PTSD label (0/1) or score
    """
    df = pd.read_csv(csv_path)
    
    # Determine audio path column
    if "audio_path" in df.columns:
        audio_path_col = "audio_path"
    elif "video_path" in df.columns:
        audio_path_col = "video_path"
    else:
        raise ValueError("CSV must contain 'audio_path' or 'video_path' column")
    
    # Prepend audio_dir if provided
    if audio_dir:
        df[audio_path_col] = df[audio_path_col].apply(
            lambda x: os.path.join(audio_dir, os.path.basename(x)) if not os.path.isabs(x) else x
        )
    
    return df, audio_path_col


def prepare_features(df: pd.DataFrame, audio_path_col: str, analyzer: MultimodalAnalyzer,
                    use_cache: bool = True):
    """
    Extract features from audio files.
    
    Args:
        df: DataFrame with audio paths
        audio_path_col: Name of column containing audio paths
        analyzer: MultimodalAnalyzer instance
        use_cache: Whether to use cached transcripts
    
    Returns:
        Tuple of (texts, prosody_features_list)
    """
    texts = []
    prosody_features_list = []
    
    cache_file = os.path.join(ARTIFACTS_DIR, "training_cache.csv")
    cache = {}
    
    # Load cache if exists
    if use_cache and os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file)
            if "id" in cache_df.columns and "transcript" in cache_df.columns:
                cache = dict(zip(cache_df["id"].astype(str), cache_df["transcript"]))
        except Exception:
            cache = {}
    
    n_total = len(df)
    print(f"Processing {n_total} audio files...")
    
    for i, row in df.iterrows():
        audio_id = str(row.get("id", i))
        audio_path = row[audio_path_col]
        
        # Check cache for transcript
        if audio_id in cache and cache[audio_id]:
            transcript = cache[audio_id]
            print(f"[{i+1}/{n_total}] Using cached transcript for {audio_id}")
        else:
            # Extract audio if needed
            if not audio_path.endswith('.wav'):
                wav_path = os.path.join(ARTIFACTS_DIR, "wav_cache", f"{audio_id}.wav")
                if not os.path.exists(wav_path):
                    extract_audio_from_call(audio_path, wav_path)
                audio_path = wav_path
            
            # Transcribe
            transcript = analyzer.transcribe(audio_path)
            cache[audio_id] = transcript
            print(f"[{i+1}/{n_total}] Transcribed {audio_id}")
        
        # Extract prosody features
        if audio_path.endswith('.wav'):
            prosody = extract_prosody_features(audio_path)
        else:
            wav_path = os.path.join(ARTIFACTS_DIR, "wav_cache", f"{audio_id}.wav")
            if not os.path.exists(wav_path):
                extract_audio_from_call(audio_path, wav_path)
            prosody = extract_prosody_features(wav_path)
        
        texts.append(transcript)
        prosody_features_list.append(prosody)
        
        # Save cache periodically
        if (i + 1) % 10 == 0:
            pd.DataFrame({
                "id": list(cache.keys()),
                "transcript": list(cache.values())
            }).to_csv(cache_file, index=False)
    
    # Final cache save
    pd.DataFrame({
        "id": list(cache.keys()),
        "transcript": list(cache.values())
    }).to_csv(cache_file, index=False)
    
    return texts, prosody_features_list


def train_phq9_classifier(texts: list, prosody_list: list, phq9_scores: np.ndarray,
                         test_size: float = 0.2):
    """Train PHQ-9 classifier."""
    print("\n" + "="*60)
    print("Training PHQ-9 Classifier")
    print("="*60)
    
    # Convert PHQ-9 scores to binary labels (>=10 = depression)
    labels = (phq9_scores >= 10).astype(int)
    
    # Split data
    Xtr, Xva, ytr, yva = train_test_split(
        list(range(len(texts))), labels,
        test_size=test_size, stratify=labels, random_state=SEED
    )
    
    train_texts = [texts[i] for i in Xtr]
    train_prosody = [prosody_list[i] for i in Xtr]
    val_texts = [texts[i] for i in Xva]
    val_prosody = [prosody_list[i] for i in Xva]
    
    # Train classifier
    clf = PHQ9Classifier()
    clf.fit(train_texts, train_prosody, ytr)
    
    # Evaluate
    val_proba = clf.predict_proba(val_texts, val_prosody)
    val_pred = val_proba[:, 1]
    
    auroc = roc_auc_score(yva, val_pred)
    auprc = average_precision_score(yva, val_pred)
    
    print(f"Validation AUROC: {auroc:.4f}")
    print(f"Validation AUPRC: {auprc:.4f}")
    
    # Save classifier
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    clf.save(os.path.join(ARTIFACTS_DIR, "phq9_classifier.pkl"))
    print(f"Saved PHQ-9 classifier to {ARTIFACTS_DIR}/phq9_classifier.pkl")
    
    return clf


def train_anxiety_classifier(texts: list, prosody_list: list, anxiety_labels: np.ndarray,
                            test_size: float = 0.2):
    """Train Anxiety classifier."""
    print("\n" + "="*60)
    print("Training Anxiety Classifier")
    print("="*60)
    
    # Ensure binary labels
    if anxiety_labels.max() > 1:
        # Assume scores, convert to binary
        threshold = np.median(anxiety_labels)
        anxiety_labels = (anxiety_labels >= threshold).astype(int)
    
    # Split data
    Xtr, Xva, ytr, yva = train_test_split(
        list(range(len(texts))), anxiety_labels,
        test_size=test_size, stratify=anxiety_labels, random_state=SEED
    )
    
    train_texts = [texts[i] for i in Xtr]
    train_prosody = [prosody_list[i] for i in Xtr]
    val_texts = [texts[i] for i in Xva]
    val_prosody = [prosody_list[i] for i in Xva]
    
    # Train classifier
    clf = AnxietyClassifier()
    clf.fit(train_texts, train_prosody, ytr)
    
    # Evaluate
    val_proba = clf.predict_proba(val_texts, val_prosody)
    val_pred = val_proba[:, 1]
    
    auroc = roc_auc_score(yva, val_pred)
    auprc = average_precision_score(yva, val_pred)
    
    print(f"Validation AUROC: {auroc:.4f}")
    print(f"Validation AUPRC: {auprc:.4f}")
    
    # Save classifier
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    clf.save(os.path.join(ARTIFACTS_DIR, "anxiety_classifier.pkl"))
    print(f"Saved Anxiety classifier to {ARTIFACTS_DIR}/anxiety_classifier.pkl")
    
    return clf


def train_ptsd_classifier(texts: list, prosody_list: list, ptsd_labels: np.ndarray,
                          test_size: float = 0.2):
    """Train PTSD classifier."""
    print("\n" + "="*60)
    print("Training PTSD Classifier")
    print("="*60)
    
    # Ensure binary labels
    if ptsd_labels.max() > 1:
        # Assume scores, convert to binary
        threshold = np.median(ptsd_labels)
        ptsd_labels = (ptsd_labels >= threshold).astype(int)
    
    # Split data
    Xtr, Xva, ytr, yva = train_test_split(
        list(range(len(texts))), ptsd_labels,
        test_size=test_size, stratify=ptsd_labels, random_state=SEED
    )
    
    train_texts = [texts[i] for i in Xtr]
    train_prosody = [prosody_list[i] for i in Xtr]
    val_texts = [texts[i] for i in Xva]
    val_prosody = [prosody_list[i] for i in Xva]
    
    # Train classifier
    clf = PTSDClassifier()
    clf.fit(train_texts, train_prosody, ytr)
    
    # Evaluate
    val_proba = clf.predict_proba(val_texts, val_prosody)
    val_pred = val_proba[:, 1]
    
    auroc = roc_auc_score(yva, val_pred)
    auprc = average_precision_score(yva, val_pred)
    
    print(f"Validation AUROC: {auroc:.4f}")
    print(f"Validation AUPRC: {auprc:.4f}")
    
    # Save classifier
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    clf.save(os.path.join(ARTIFACTS_DIR, "ptsd_classifier.pkl"))
    print(f"Saved PTSD classifier to {ARTIFACTS_DIR}/ptsd_classifier.pkl")
    
    return clf


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train mental health classifiers")
    parser.add_argument("--data", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audio files")
    parser.add_argument("--train_phq9", action="store_true", help="Train PHQ-9 classifier")
    parser.add_argument("--train_anxiety", action="store_true", help="Train Anxiety classifier")
    parser.add_argument("--train_ptsd", action="store_true", help="Train PTSD classifier")
    parser.add_argument("--train_all", action="store_true", help="Train all classifiers")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    df, audio_path_col = load_training_data(args.data, args.audio_dir)
    print(f"Loaded {len(df)} samples")
    
    # Initialize analyzer
    print("\nInitializing multimodal analyzer...")
    analyzer = MultimodalAnalyzer()
    
    # Extract features
    print("\nExtracting features...")
    texts, prosody_list = prepare_features(df, audio_path_col, analyzer)
    
    # Train classifiers
    if args.train_all or args.train_phq9:
        if "phq9_total" not in df.columns:
            print("Warning: 'phq9_total' column not found. Skipping PHQ-9 training.")
        else:
            train_phq9_classifier(texts, prosody_list, df["phq9_total"].values)
    
    if args.train_all or args.train_anxiety:
        if "anxiety_label" not in df.columns:
            print("Warning: 'anxiety_label' column not found. Skipping Anxiety training.")
        else:
            train_anxiety_classifier(texts, prosody_list, df["anxiety_label"].values)
    
    if args.train_all or args.train_ptsd:
        if "ptsd_label" not in df.columns:
            print("Warning: 'ptsd_label' column not found. Skipping PTSD training.")
        else:
            train_ptsd_classifier(texts, prosody_list, df["ptsd_label"].values)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
