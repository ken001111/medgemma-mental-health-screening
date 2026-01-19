"""
MedGemma integration for medical text analysis and clinical reasoning.
Enhances transcript analysis with Google's MedGemma medical language model.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from config import MEDGEMMA_MODEL_ID, MEDGEMMA_MAX_TOKENS, USE_MEDGEMMA_27B_FOR_CRITICAL


class MedGemmaAnalyzer:
    """
    Analyzes medical transcripts using MedGemma for clinical insights.
    """
    
    def __init__(self, model_id: str = None, device: str = None):
        """
        Initialize MedGemma analyzer.
        
        Args:
            model_id: MedGemma model ID (defaults to config)
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_id = model_id or MEDGEMMA_MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading MedGemma model ({self.model_id}) on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            self.model.eval()
            print(f"MedGemma model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load MedGemma model: {e}")
            print("MedGemma features will be disabled. Falling back to basic analysis.")
            self.model = None
            self.tokenizer = None
    
    def is_available(self) -> bool:
        """Check if MedGemma is available."""
        return self.model is not None and self.tokenizer is not None
    
    @torch.no_grad()
    def analyze_transcript(self, transcript: str, scores: Dict[str, float] = None) -> Dict[str, any]:
        """
        Analyze transcript for medical insights using MedGemma.
        
        Args:
            transcript: Transcribed text from phone call
            scores: Optional screening scores (PHQ-9, anxiety, PTSD)
        
        Returns:
            Dictionary with medical insights and analysis
        """
        if not self.is_available():
            return {
                "medical_insights": [],
                "risk_indicators": [],
                "clinical_summary": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                "medgemma_available": False
            }
        
        try:
            # Build prompt for medical analysis
            prompt = self._build_analysis_prompt(transcript, scores)
            
            # Generate analysis
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MEDGEMMA_MAX_TOKENS,
                temperature=0.3,  # Lower temperature for more focused analysis
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response into structured insights
            insights = self._parse_medgemma_response(response, transcript, scores)
            
            return insights
        
        except Exception as e:
            print(f"Error in MedGemma analysis: {e}")
            return {
                "medical_insights": [],
                "risk_indicators": [],
                "clinical_summary": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                "medgemma_available": False,
                "error": str(e)
            }
    
    def _build_analysis_prompt(self, transcript: str, scores: Dict[str, float] = None) -> str:
        """Build prompt for MedGemma analysis."""
        
        prompt = """You are a medical AI assistant analyzing a mental health screening phone call transcript. 
Provide a clinical analysis focusing on mental health indicators, risk factors, and recommendations.

TRANSCRIPT:
"""
        prompt += transcript[:2000]  # Limit transcript length
        
        if scores:
            prompt += "\n\nSCREENING SCORES:\n"
            if scores.get("phq9_score") is not None:
                prompt += f"PHQ-9 (Depression): {scores['phq9_score']:.1f}/27\n"
            if scores.get("anxiety_risk") is not None:
                prompt += f"Anxiety Risk: {scores['anxiety_risk']:.2%}\n"
            if scores.get("ptsd_risk") is not None:
                prompt += f"PTSD Risk: {scores['ptsd_risk']:.2%}\n"
        
        prompt += """
\nPlease provide:
1. Key medical insights and observations
2. Mental health risk indicators identified
3. Clinical summary (2-3 sentences)
4. Recommended follow-up actions

ANALYSIS:
"""
        
        return prompt
    
    def _parse_medgemma_response(self, response: str, transcript: str, scores: Dict[str, float]) -> Dict[str, any]:
        """Parse MedGemma response into structured format."""
        
        # Extract insights (simple parsing - can be enhanced)
        insights = []
        risk_indicators = []
        
        # Look for key phrases in response
        response_lower = response.lower()
        
        # Extract risk indicators
        risk_keywords = [
            "suicide", "self-harm", "hopeless", "worthless",
            "trauma", "flashback", "nightmare", "avoidance",
            "panic", "anxiety", "worry", "fear",
            "depression", "sad", "empty", "loss of interest"
        ]
        
        for keyword in risk_keywords:
            if keyword in response_lower:
                risk_indicators.append(keyword)
        
        # Extract insights (split by common delimiters)
        if "1." in response or "insights" in response_lower:
            # Try to extract numbered or bulleted items
            lines = response.split('\n')
            for line in lines:
                if any(marker in line for marker in ['•', '-', '1.', '2.', '3.']):
                    cleaned = line.strip().lstrip('•-1234567890. ').strip()
                    if cleaned and len(cleaned) > 10:
                        insights.append(cleaned)
        
        # If no structured insights found, use first few sentences
        if not insights:
            sentences = response.split('.')[:3]
            insights = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return {
            "medical_insights": insights[:5],  # Limit to top 5
            "risk_indicators": list(set(risk_indicators)),  # Remove duplicates
            "clinical_summary": response[:500] if len(response) > 500 else response,
            "full_analysis": response,
            "medgemma_available": True
        }
    
    @torch.no_grad()
    def generate_clinical_summary(self, transcript: str, scores: Dict[str, float]) -> str:
        """
        Generate a clinical summary using MedGemma.
        
        Args:
            transcript: Transcribed text
            scores: Screening scores
        
        Returns:
            Clinical summary text
        """
        if not self.is_available():
            return self._generate_basic_summary(transcript, scores)
        
        try:
            prompt = f"""Generate a concise clinical summary for a mental health screening call.

TRANSCRIPT:
{transcript[:1500]}

SCORES:
"""
            if scores.get("phq9_score") is not None:
                prompt += f"PHQ-9: {scores['phq9_score']:.1f}/27\n"
            if scores.get("anxiety_risk") is not None:
                prompt += f"Anxiety Risk: {scores['anxiety_risk']:.2%}\n"
            if scores.get("ptsd_risk") is not None:
                prompt += f"PTSD Risk: {scores['ptsd_risk']:.2%}\n"
            
            prompt += "\nCLINICAL SUMMARY:\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.4,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            summary = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return summary.strip()
        
        except Exception as e:
            print(f"Error generating clinical summary: {e}")
            return self._generate_basic_summary(transcript, scores)
    
    def _generate_basic_summary(self, transcript: str, scores: Dict[str, float]) -> str:
        """Generate basic summary without MedGemma."""
        summary = "Patient reported symptoms during screening call. "
        
        if scores.get("phq9_score") is not None:
            phq9 = scores["phq9_score"]
            if phq9 >= 15:
                summary += f"Severe depression indicated (PHQ-9: {phq9:.1f}). "
            elif phq9 >= 10:
                summary += f"Moderate depression indicated (PHQ-9: {phq9:.1f}). "
        
        if scores.get("anxiety_risk") is not None and scores["anxiety_risk"] >= 0.6:
            summary += f"Elevated anxiety risk ({scores['anxiety_risk']:.2%}). "
        
        if scores.get("ptsd_risk") is not None and scores["ptsd_risk"] >= 0.6:
            summary += f"Elevated PTSD risk ({scores['ptsd_risk']:.2%}). "
        
        summary += "Clinical review recommended."
        return summary
    
    @torch.no_grad()
    def detect_suicide_risk(self, transcript: str) -> Dict[str, any]:
        """
        Use MedGemma to detect suicide risk with clinical reasoning.
        
        Args:
            transcript: Transcribed text
        
        Returns:
            Dictionary with risk assessment
        """
        if not self.is_available():
            # Fallback to keyword matching
            transcript_lower = transcript.lower()
            keywords_found = [kw for kw in ["suicide", "kill myself", "end it all"] if kw in transcript_lower]
            return {
                "risk_detected": len(keywords_found) > 0,
                "confidence": 0.5 if keywords_found else 0.1,
                "indicators": keywords_found,
                "reasoning": "Keyword-based detection (MedGemma unavailable)"
            }
        
        try:
            prompt = f"""Analyze this transcript for suicide risk indicators. Be clinically precise.

TRANSCRIPT:
{transcript[:1500]}

Provide:
1. Risk level (none/low/moderate/high/critical)
2. Specific indicators found
3. Clinical reasoning

ANALYSIS:
"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.2,  # Very low temperature for critical assessment
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse risk level
            response_lower = response.lower()
            risk_level = "low"
            if "critical" in response_lower or "immediate" in response_lower:
                risk_level = "critical"
            elif "high" in response_lower:
                risk_level = "high"
            elif "moderate" in response_lower:
                risk_level = "moderate"
            
            confidence = 0.9 if risk_level in ["critical", "high"] else 0.6
            
            return {
                "risk_detected": risk_level in ["moderate", "high", "critical"],
                "risk_level": risk_level,
                "confidence": confidence,
                "indicators": self._extract_indicators(response),
                "reasoning": response[:300],
                "full_analysis": response
            }
        
        except Exception as e:
            print(f"Error in suicide risk detection: {e}")
            return {
                "risk_detected": False,
                "confidence": 0.0,
                "indicators": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def _extract_indicators(self, text: str) -> List[str]:
        """Extract risk indicators from text."""
        indicators = []
        text_lower = text.lower()
        
        indicator_phrases = [
            "suicidal ideation", "suicide plan", "suicide attempt",
            "hopelessness", "worthlessness", "no reason to live",
            "self-harm", "ending it", "better off dead"
        ]
        
        for phrase in indicator_phrases:
            if phrase in text_lower:
                indicators.append(phrase)
        
        return indicators


# Factory function for creating analyzer with appropriate model
def create_medgemma_analyzer(use_27b: bool = False, for_critical_case: bool = False) -> MedGemmaAnalyzer:
    """
    Create MedGemma analyzer with appropriate model.
    
    Args:
        use_27b: Force use of 27B model
        for_critical_case: If True and USE_MEDGEMMA_27B_FOR_CRITICAL is enabled, use 27B
    
    Returns:
        MedGemmaAnalyzer instance
    """
    if use_27b or (for_critical_case and USE_MEDGEMMA_27B_FOR_CRITICAL):
        model_id = "google/medgemma-2-27b-it"
    else:
        model_id = MEDGEMMA_MODEL_ID
    
    return MedGemmaAnalyzer(model_id=model_id)
