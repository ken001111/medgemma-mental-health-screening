"""
Multimodal analysis module combining text (transcription) and prosody features.
Uses Google Health AI models for medical speech recognition and analysis.
"""
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Dict, Tuple, Optional
from config import MEDASR_ID, MAX_NEW_TOKENS, SR
from audio_processor import extract_prosody_features, load_audio
from medgemma_analyzer import MedGemmaAnalyzer, create_medgemma_analyzer


class MultimodalAnalyzer:
    """
    Analyzes phone calls using both text (transcription) and prosody features.
    """
    
    def __init__(self, device: str = None, use_medgemma: bool = True):
        """
        Initialize the multimodal analyzer with MedASR and MedGemma models.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            use_medgemma: Whether to initialize MedGemma for text analysis
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading MedASR model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(MEDASR_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MEDASR_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print("MedASR model loaded successfully.")
        
        # Initialize MedGemma for medical text analysis
        self.medgemma = None
        if use_medgemma:
            try:
                self.medgemma = create_medgemma_analyzer(use_27b=False)
            except Exception as e:
                print(f"Warning: Could not initialize MedGemma: {e}")
                print("Continuing without MedGemma analysis.")
    
    @torch.no_grad()
    def transcribe(self, wav_path: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        Transcribe audio using MedASR model.
        
        Args:
            wav_path: Path to WAV audio file
            max_new_tokens: Maximum tokens for generation
        
        Returns:
            Transcribed text
        """
        try:
            audio, sr = load_audio(wav_path)
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
            
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            
            transcript = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            return transcript
        
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
    
    def analyze_call(self, wav_path: str, scores: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Perform multimodal analysis on a phone call.
        Combines text transcription, prosody features, and MedGemma medical analysis.
        
        Args:
            wav_path: Path to WAV audio file
            scores: Optional screening scores for MedGemma analysis
        
        Returns:
            Dictionary containing:
            - transcript: Transcribed text
            - prosody_features: Dictionary of prosody features
            - multimodal_features: Combined feature vector for classification
            - medgemma_analysis: Medical insights from MedGemma (if available)
        """
        # Extract text (transcription)
        transcript = self.transcribe(wav_path)
        
        # Extract prosody features
        prosody_features = extract_prosody_features(wav_path)
        
        # Combine features for multimodal analysis
        multimodal_features = self._combine_features(transcript, prosody_features)
        
        result = {
            "transcript": transcript,
            "prosody_features": prosody_features,
            "multimodal_features": multimodal_features,
        }
        
        # Add MedGemma analysis if available
        if self.medgemma and self.medgemma.is_available():
            medgemma_analysis = self.medgemma.analyze_transcript(transcript, scores)
            result["medgemma_analysis"] = medgemma_analysis
        
        return result
    
    def _combine_features(self, transcript: str, prosody_features: Dict[str, float]) -> Dict[str, any]:
        """
        Combine text and prosody features into a unified representation.
        
        Args:
            transcript: Transcribed text
            prosody_features: Dictionary of prosody features
        
        Returns:
            Combined feature dictionary
        """
        # For now, return both separately. Classifiers will handle combination.
        # In a more advanced implementation, you might use a multimodal encoder.
        return {
            "text": transcript,
            "prosody": prosody_features,
            "text_length": len(transcript.split()),
            "has_speech": len(transcript.strip()) > 0,
        }
