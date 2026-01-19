"""
Medical report generation module for clinician review.
"""
from datetime import datetime
from typing import Dict, Optional
import os
from config import REPORTS_DIR, USE_MEDGEMMA_27B_FOR_CRITICAL
from medgemma_analyzer import create_medgemma_analyzer


class ReportGenerator:
    """
    Generates medical reports for clinician review.
    """
    
    def __init__(self, use_medgemma: bool = True):
        """
        Initialize report generator.
        
        Args:
            use_medgemma: Whether to use MedGemma for enhanced report generation
        """
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.use_medgemma = use_medgemma
        self.medgemma_analyzer = None
        
        if use_medgemma:
            try:
                self.medgemma_analyzer = create_medgemma_analyzer(use_27b=False)
            except Exception as e:
                print(f"Warning: Could not initialize MedGemma for reports: {e}")
                self.use_medgemma = False
    
    def generate_report(self, call_id: str, soldier_id: str, call_timestamp: datetime,
                       transcript: str, scores: Dict[str, float],
                       risk_assessment: Dict[str, any],
                       prosody_features: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Generate a medical report for a screening call.
        
        Args:
            call_id: Unique call identifier
            soldier_id: Soldier identifier
            call_timestamp: Timestamp of the call
            transcript: Transcribed text from call
            scores: Dictionary with phq9_score, anxiety_risk, ptsd_risk
            risk_assessment: Risk assessment dictionary
        
        Returns:
            Dictionary with report_content and report_path
        """
        # Format report content
        report_content = self._format_report(
            call_id, soldier_id, call_timestamp,
            transcript, scores, risk_assessment, prosody_features
        )
        
        # Save report to file
        report_filename = f"report_{call_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return {
            "report_content": report_content,
            "report_path": report_path
        }
    
    def _format_report(self, call_id: str, soldier_id: str, call_timestamp: datetime,
                      transcript: str, scores: Dict[str, float],
                      risk_assessment: Dict[str, any],
                      prosody_features: Optional[Dict[str, float]] = None) -> str:
        """Format the report content."""
        
        # Format scores section
        scores_section = "SCREENING SCORES:\n"
        scores_section += "=" * 50 + "\n"
        if scores.get("phq9_score") is not None:
            phq9 = scores["phq9_score"]
            severity = self._get_phq9_severity(phq9)
            scores_section += f"PHQ-9 Score: {phq9:.1f}/27 ({severity})\n"
        if scores.get("anxiety_risk") is not None:
            anxiety = scores["anxiety_risk"]
            severity = self._get_risk_severity(anxiety, "anxiety")
            scores_section += f"Anxiety Risk: {anxiety:.2%} ({severity})\n"
        if scores.get("ptsd_risk") is not None:
            ptsd = scores["ptsd_risk"]
            severity = self._get_risk_severity(ptsd, "ptsd")
            scores_section += f"PTSD Risk: {ptsd:.2%} ({severity})\n"
        scores_section += "\n"
        
        # Format risk assessment section
        risk_section = "RISK ASSESSMENT:\n"
        risk_section += "=" * 50 + "\n"
        risk_section += f"Overall Severity: {risk_assessment['severity'].upper()}\n"
        if risk_assessment.get("risk_factors"):
            risk_section += "\nRisk Factors Identified:\n"
            for factor in risk_assessment["risk_factors"]:
                risk_section += f"  - {factor}\n"
        if risk_assessment.get("alerts"):
            risk_section += "\nActive Alerts:\n"
            for alert in risk_assessment["alerts"]:
                risk_section += f"  - [{alert['severity'].upper()}] {alert['message']}\n"
        risk_section += "\n"
        
        # Format transcript section
        transcript_section = "CALL TRANSCRIPT:\n"
        transcript_section += "=" * 50 + "\n"
        transcript_section += transcript if transcript else "[No transcript available]\n"
        transcript_section += "\n"

        # Format prosody section
        prosody_section = "PROSODY FEATURES:\n"
        prosody_section += "=" * 50 + "\n"
        if prosody_features:
            duration = prosody_features.get("duration", 0.0)
            mean_pitch = prosody_features.get("mean_pitch", 0.0)
            pitch_std = prosody_features.get("pitch_std", 0.0)
            speaking_rate = prosody_features.get("speaking_rate", 0.0)
            mean_energy = prosody_features.get("mean_energy", 0.0)
            energy_std = prosody_features.get("energy_std", 0.0)
            silence_ratio = prosody_features.get("silence_ratio", 0.0)
            mean_spectral_centroid = prosody_features.get("mean_spectral_centroid", 0.0)
            jitter = prosody_features.get("jitter", 0.0)
            prosody_section += f"Duration: {duration:.2f} sec\n"
            prosody_section += f"Mean Pitch: {mean_pitch:.2f} Hz\n"
            prosody_section += f"Pitch Std: {pitch_std:.2f} Hz\n"
            prosody_section += f"Speaking Rate (ZCR proxy): {speaking_rate:.2f}\n"
            prosody_section += f"Mean Energy: {mean_energy:.4f}\n"
            prosody_section += f"Energy Std: {energy_std:.4f}\n"
            prosody_section += f"Silence Ratio: {silence_ratio:.2%}\n"
            prosody_section += f"Mean Spectral Centroid: {mean_spectral_centroid:.2f}\n"
            prosody_section += f"Jitter (proxy): {jitter:.4f}\n"
        else:
            prosody_section += "[No prosody features available]\n"
        prosody_section += "\n"
        
        # Add MedGemma clinical analysis if available
        medgemma_section = ""
        if self.use_medgemma and self.medgemma_analyzer and self.medgemma_analyzer.is_available():
            try:
                # Use 27B for high-risk cases if configured
                is_high_risk = risk_assessment.get("severity") in ["high", "critical"]
                if is_high_risk and USE_MEDGEMMA_27B_FOR_CRITICAL:
                    medgemma_27b = create_medgemma_analyzer(use_27b=True, for_critical_case=True)
                    clinical_summary = medgemma_27b.generate_clinical_summary(transcript, scores)
                    medgemma_analysis = medgemma_27b.analyze_transcript(transcript, scores)
                else:
                    clinical_summary = self.medgemma_analyzer.generate_clinical_summary(transcript, scores)
                    medgemma_analysis = self.medgemma_analyzer.analyze_transcript(transcript, scores)
                
                medgemma_section = "CLINICAL ANALYSIS (MedGemma):\n"
                medgemma_section += "=" * 50 + "\n"
                medgemma_section += f"{clinical_summary}\n\n"
                
                if medgemma_analysis.get("medical_insights"):
                    medgemma_section += "Key Medical Insights:\n"
                    for insight in medgemma_analysis["medical_insights"]:
                        medgemma_section += f"  - {insight}\n"
                    medgemma_section += "\n"
                
                if medgemma_analysis.get("risk_indicators"):
                    medgemma_section += "Risk Indicators Identified:\n"
                    for indicator in medgemma_analysis["risk_indicators"]:
                        medgemma_section += f"  - {indicator}\n"
                    medgemma_section += "\n"
            except Exception as e:
                print(f"Error generating MedGemma analysis: {e}")
                medgemma_section = ""
        
        # Compile full report
        report = f"""
MENTAL HEALTH SCREENING REPORT
{'=' * 50}

CALL INFORMATION:
Call ID: {call_id}
Soldier ID: {soldier_id}
Call Date/Time: {call_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{scores_section}
{risk_section}
{prosody_section}
{medgemma_section}
{transcript_section}

CLINICIAN NOTES:
[Space for clinician notes and recommendations]

{'=' * 50}
This is an automated screening report. Clinical judgment is required for final assessment.
"""
        
        return report
    
    def _get_phq9_severity(self, score: float) -> str:
        """Get PHQ-9 severity level."""
        if score >= 20:
            return "Severe"
        elif score >= 15:
            return "Moderately Severe"
        elif score >= 10:
            return "Moderate"
        elif score >= 5:
            return "Mild"
        else:
            return "Minimal"
    
    def _get_risk_severity(self, risk: float, condition: str) -> str:
        """Get risk severity level."""
        if risk >= 0.8:
            return "Severe"
        elif risk >= 0.6:
            return "Moderate"
        elif risk >= 0.4:
            return "Mild"
        else:
            return "Low"
