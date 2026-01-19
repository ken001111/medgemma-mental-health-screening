"""
Alert system for high-risk cases requiring immediate attention.
"""
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from config import (
    SUICIDE_RISK_KEYWORDS, CRITICAL_RISK_THRESHOLD,
    PHQ9_THRESHOLD_SEVERE, ANXIETY_THRESHOLD_SEVERE, PTSD_THRESHOLD_SEVERE,
    ALERT_EMAILS, ALERT_PHONE_NUMBERS, USE_MEDGEMMA_27B_FOR_CRITICAL
)
from database import ScreeningDatabase
from medgemma_analyzer import create_medgemma_analyzer


class AlertSystem:
    """
    Monitors screening results and generates alerts for high-risk cases.
    """
    
    def __init__(self, database: ScreeningDatabase, use_medgemma: bool = True):
        """
        Initialize alert system.
        
        Args:
            database: Database instance for storing alerts
            use_medgemma: Whether to use MedGemma for enhanced risk detection
        """
        self.database = database
        self.use_medgemma = use_medgemma
        self.medgemma_analyzer = None
        
        if use_medgemma:
            try:
                # Start with 4B, will upgrade to 27B for critical cases if needed
                self.medgemma_analyzer = create_medgemma_analyzer(use_27b=False)
            except Exception as e:
                print(f"Warning: Could not initialize MedGemma for alerts: {e}")
                self.use_medgemma = False
    
    def check_suicide_risk(self, transcript: str, use_medgemma: bool = True) -> Tuple[bool, str]:
        """
        Check transcript for suicide risk indicators.
        Uses MedGemma for enhanced detection if available.
        
        Args:
            transcript: Transcribed text from call
            use_medgemma: Whether to use MedGemma for analysis
        
        Returns:
            Tuple of (is_risk, risk_message)
        """
        # First check with keyword matching
        transcript_lower = transcript.lower()
        keyword_found = None
        
        for keyword in SUICIDE_RISK_KEYWORDS:
            if keyword.lower() in transcript_lower:
                keyword_found = keyword
                break
        
        # If keyword found or MedGemma available, use MedGemma for deeper analysis
        if (keyword_found or use_medgemma) and self.use_medgemma and self.medgemma_analyzer:
            try:
                # For critical cases, use 27B if configured
                is_critical = keyword_found is not None
                if is_critical and USE_MEDGEMMA_27B_FOR_CRITICAL:
                    # Upgrade to 27B for critical analysis
                    medgemma_27b = create_medgemma_analyzer(use_27b=True, for_critical_case=True)
                    risk_assessment = medgemma_27b.detect_suicide_risk(transcript)
                else:
                    risk_assessment = self.medgemma_analyzer.detect_suicide_risk(transcript)
                
                if risk_assessment.get("risk_detected"):
                    risk_level = risk_assessment.get("risk_level", "unknown")
                    reasoning = risk_assessment.get("reasoning", "")
                    return True, f"Suicide risk detected ({risk_level}): {reasoning[:200]}"
            except Exception as e:
                print(f"Error in MedGemma suicide risk detection: {e}")
                # Fall back to keyword matching
        
        # Fallback to keyword matching
        if keyword_found:
            return True, f"Suicide risk detected: keyword '{keyword_found}' found in transcript"
        
        return False, ""
    
    def assess_risk_level(self, phq9_score: float = None, anxiety_risk: float = None,
                         ptsd_risk: float = None, transcript: str = "") -> Dict[str, any]:
        """
        Assess overall risk level based on scores and transcript.
        
        Args:
            phq9_score: PHQ-9 depression score
            anxiety_risk: Anxiety risk probability
            ptsd_risk: PTSD risk probability
            transcript: Transcribed text
        
        Returns:
            Dictionary with risk assessment details
        """
        alerts = []
        severity = "low"
        risk_factors = []
        
        # Check for suicide risk (highest priority)
        suicide_risk, suicide_message = self.check_suicide_risk(transcript)
        if suicide_risk:
            alerts.append({
                "type": "suicide_risk",
                "severity": "critical",
                "message": suicide_message
            })
            severity = "critical"
            risk_factors.append("Suicide risk indicators")
        
        # Check PHQ-9 score
        if phq9_score is not None:
            if phq9_score >= PHQ9_THRESHOLD_SEVERE:
                alerts.append({
                    "type": "depression",
                    "severity": "high",
                    "message": f"Severe depression detected: PHQ-9 score = {phq9_score:.1f}"
                })
                if severity != "critical":
                    severity = "high"
                risk_factors.append(f"Severe depression (PHQ-9: {phq9_score:.1f})")
            elif phq9_score >= 10:
                risk_factors.append(f"Moderate depression (PHQ-9: {phq9_score:.1f})")
                if severity == "low":
                    severity = "moderate"
        
        # Check anxiety risk
        if anxiety_risk is not None:
            if anxiety_risk >= ANXIETY_THRESHOLD_SEVERE:
                alerts.append({
                    "type": "anxiety",
                    "severity": "high",
                    "message": f"Severe anxiety detected: risk probability = {anxiety_risk:.2f}"
                })
                if severity != "critical":
                    severity = "high"
                risk_factors.append(f"Severe anxiety (risk: {anxiety_risk:.2f})")
            elif anxiety_risk >= 0.6:
                risk_factors.append(f"Moderate anxiety (risk: {anxiety_risk:.2f})")
                if severity == "low":
                    severity = "moderate"
        
        # Check PTSD risk
        if ptsd_risk is not None:
            if ptsd_risk >= PTSD_THRESHOLD_SEVERE:
                alerts.append({
                    "type": "ptsd",
                    "severity": "high",
                    "message": f"Severe PTSD detected: risk probability = {ptsd_risk:.2f}"
                })
                if severity != "critical":
                    severity = "high"
                risk_factors.append(f"Severe PTSD (risk: {ptsd_risk:.2f})")
            elif ptsd_risk >= 0.6:
                risk_factors.append(f"Moderate PTSD (risk: {ptsd_risk:.2f})")
                if severity == "low":
                    severity = "moderate"
        
        return {
            "severity": severity,
            "alerts": alerts,
            "risk_factors": risk_factors,
            "requires_immediate_attention": severity in ["critical", "high"]
        }
    
    def generate_alert(self, call_id: str, assessment: Dict[str, any]) -> List[Dict]:
        """
        Generate and store alerts based on risk assessment.
        
        Args:
            call_id: ID of the call
            assessment: Risk assessment dictionary
        
        Returns:
            List of generated alert records
        """
        generated_alerts = []
        
        for alert_info in assessment["alerts"]:
            # Determine recipients based on severity
            recipients = []
            if alert_info["severity"] == "critical":
                recipients = ALERT_EMAILS + ALERT_PHONE_NUMBERS
            elif alert_info["severity"] == "high":
                recipients = ALERT_EMAILS
            
            sent_to = ", ".join(recipients) if recipients else None
            
            # Store alert in database
            self.database.add_alert(
                call_id=call_id,
                alert_type=alert_info["type"],
                severity=alert_info["severity"],
                message=alert_info["message"],
                sent_to=sent_to
            )
            
            generated_alerts.append({
                "call_id": call_id,
                "type": alert_info["type"],
                "severity": alert_info["severity"],
                "message": alert_info["message"],
                "sent_to": sent_to
            })
            
            # In production, send actual notifications here
            # self._send_notification(alert_info, recipients)
        
        return generated_alerts
    
    def _send_notification(self, alert_info: Dict, recipients: List[str]):
        """
        Send notification to recipients (to be implemented with actual notification service).
        
        Args:
            alert_info: Alert information dictionary
            recipients: List of recipient contact information
        """
        # TODO: Implement actual notification sending
        # - Email via SMTP
        # - SMS via Twilio or similar
        # - Push notifications
        print(f"[ALERT] {alert_info['severity'].upper()}: {alert_info['message']}")
        print(f"Recipients: {recipients}")
