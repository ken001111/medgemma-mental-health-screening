"""
Main application for mental health screening system.
Processes phone calls, performs multimodal analysis, and generates reports.
"""
import os
import uuid
from datetime import datetime
from typing import Dict, Optional
from multimodal_analyzer import MultimodalAnalyzer
from classifiers import PHQ9Classifier, AnxietyClassifier, PTSDClassifier
from database import ScreeningDatabase
from alert_system import AlertSystem
from report_generator import ReportGenerator
from audio_processor import extract_audio_from_call
from config import ARTIFACTS_DIR


class MentalHealthScreeningApp:
    """
    Main application class for processing phone calls and generating screening results.
    """
    
    def __init__(self, load_classifiers: bool = True):
        """
        Initialize the screening application.
        
        Args:
            load_classifiers: Whether to load pre-trained classifiers from disk
        """
        print("Initializing Mental Health Screening Application...")
        
        # Initialize components
        self.analyzer = MultimodalAnalyzer()
        self.database = ScreeningDatabase()
        self.alert_system = AlertSystem(self.database)
        self.report_generator = ReportGenerator()
        
        # Initialize classifiers
        self.phq9_classifier = PHQ9Classifier()
        self.anxiety_classifier = AnxietyClassifier()
        self.ptsd_classifier = PTSDClassifier()
        
        # Load classifiers if available
        if load_classifiers:
            self._load_classifiers()
        
        print("Application initialized successfully.")
    
    def _load_classifiers(self):
        """Load pre-trained classifiers from disk if available."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        phq9_path = os.path.join(ARTIFACTS_DIR, "phq9_classifier.pkl")
        anxiety_path = os.path.join(ARTIFACTS_DIR, "anxiety_classifier.pkl")
        ptsd_path = os.path.join(ARTIFACTS_DIR, "ptsd_classifier.pkl")
        
        if os.path.exists(phq9_path):
            try:
                self.phq9_classifier.load(phq9_path)
                print("Loaded PHQ-9 classifier.")
            except Exception as e:
                print(f"Warning: Could not load PHQ-9 classifier: {e}")
        
        if os.path.exists(anxiety_path):
            try:
                self.anxiety_classifier.load(anxiety_path)
                print("Loaded Anxiety classifier.")
            except Exception as e:
                print(f"Warning: Could not load Anxiety classifier: {e}")
        
        if os.path.exists(ptsd_path):
            try:
                self.ptsd_classifier.load(ptsd_path)
                print("Loaded PTSD classifier.")
            except Exception as e:
                print(f"Warning: Could not load PTSD classifier: {e}")
    
    def process_call(self, call_path: str, soldier_id: str,
                    call_timestamp: Optional[datetime] = None) -> Dict[str, any]:
        """
        Process a phone call through the complete screening pipeline.
        
        Args:
            call_path: Path to the phone call audio/video file
            soldier_id: Identifier for the soldier
            call_timestamp: Timestamp of the call (defaults to now)
        
        Returns:
            Dictionary with complete screening results
        """
        if call_timestamp is None:
            call_timestamp = datetime.now()
        
        # Generate unique call ID
        call_id = str(uuid.uuid4())
        
        print(f"\nProcessing call {call_id} for soldier {soldier_id}...")
        
        try:
            # Step 1: Extract audio from call
            print("Step 1: Extracting audio...")
            wav_path = extract_audio_from_call(call_path)
            
            # Step 2: Multimodal analysis (text + prosody)
            print("Step 2: Performing multimodal analysis...")
            analysis = self.analyzer.analyze_call(wav_path)
            transcript = analysis["transcript"]
            prosody_features = analysis["prosody_features"]
            
            # Step 3: Run classifiers
            print("Step 3: Running mental health classifiers...")
            scores = {}
            
            if self.phq9_classifier.is_fitted:
                phq9_score = self.phq9_classifier.predict_phq9_score(
                    [transcript], [prosody_features]
                )
                scores["phq9_score"] = phq9_score
                print(f"  PHQ-9 Score: {phq9_score:.1f}/27")
            
            if self.anxiety_classifier.is_fitted:
                anxiety_risk = self.anxiety_classifier.predict_anxiety_risk(
                    [transcript], [prosody_features]
                )
                scores["anxiety_risk"] = anxiety_risk
                print(f"  Anxiety Risk: {anxiety_risk:.2%}")
            
            if self.ptsd_classifier.is_fitted:
                ptsd_risk = self.ptsd_classifier.predict_ptsd_risk(
                    [transcript], [prosody_features]
                )
                scores["ptsd_risk"] = ptsd_risk
                print(f"  PTSD Risk: {ptsd_risk:.2%}")
            
            # Step 2.5: Enhanced analysis with MedGemma (now that scores are available)
            if self.analyzer.medgemma and self.analyzer.medgemma.is_available():
                print("Step 2.5: Enhanced MedGemma analysis with scores...")
                medgemma_analysis = self.analyzer.medgemma.analyze_transcript(transcript, scores)
                analysis["medgemma_analysis"] = medgemma_analysis
                if medgemma_analysis.get("medical_insights"):
                    print(f"  Found {len(medgemma_analysis['medical_insights'])} medical insights")
            
            # Step 4: Risk assessment and alert generation
            print("Step 4: Assessing risk and generating alerts...")
            risk_assessment = self.alert_system.assess_risk_level(
                phq9_score=scores.get("phq9_score"),
                anxiety_risk=scores.get("anxiety_risk"),
                ptsd_risk=scores.get("ptsd_risk"),
                transcript=transcript
            )
            
            # Generate alerts if needed
            alerts = []
            if risk_assessment["requires_immediate_attention"]:
                alerts = self.alert_system.generate_alert(call_id, risk_assessment)
                print(f"  Generated {len(alerts)} alert(s)")
            
            # Step 5: Save to database
            print("Step 5: Saving to database...")
            call_duration = prosody_features.get("duration", 0.0)
            self.database.add_call(
                call_id=call_id,
                soldier_id=soldier_id,
                call_timestamp=call_timestamp,
                call_duration=call_duration,
                audio_path=wav_path,
                transcript=transcript
            )
            
            self.database.add_scores(
                call_id=call_id,
                phq9_score=scores.get("phq9_score"),
                anxiety_risk=scores.get("anxiety_risk"),
                ptsd_risk=scores.get("ptsd_risk")
            )
            
            # Step 6: Generate medical report
            print("Step 6: Generating medical report...")
            report = self.report_generator.generate_report(
                call_id=call_id,
                soldier_id=soldier_id,
                call_timestamp=call_timestamp,
                transcript=transcript,
                scores=scores,
                risk_assessment=risk_assessment
            )
            
            self.database.add_report(
                call_id=call_id,
                report_content=report["report_content"],
                report_path=report["report_path"]
            )
            
            print(f"\n✓ Call processing complete!")
            print(f"  Report saved to: {report['report_path']}")
            if alerts:
                print(f"  ⚠ {len(alerts)} alert(s) generated - immediate attention required!")
            
            return {
                "call_id": call_id,
                "soldier_id": soldier_id,
                "scores": scores,
                "risk_assessment": risk_assessment,
                "alerts": alerts,
                "report_path": report["report_path"],
                "transcript": transcript
            }
        
        except Exception as e:
            print(f"\n✗ Error processing call: {e}")
            raise
    
    def get_soldier_history(self, soldier_id: str, limit: int = 10) -> list:
        """Get screening history for a soldier."""
        return self.database.get_call_history(soldier_id, limit)
    
    def get_pending_alerts(self) -> list:
        """Get all pending alerts."""
        return self.database.get_pending_alerts()


def main():
    """Example usage of the screening application."""
    app = MentalHealthScreeningApp()
    
    # Example: Process a call
    # Replace with actual call file path
    example_call_path = "example_call.wav"
    
    if os.path.exists(example_call_path):
        result = app.process_call(
            call_path=example_call_path,
            soldier_id="SOLDIER_001"
        )
        print("\nScreening Results:")
        print(f"  Call ID: {result['call_id']}")
        print(f"  Scores: {result['scores']}")
        print(f"  Risk Level: {result['risk_assessment']['severity']}")
    else:
        print(f"Example call file not found: {example_call_path}")
        print("Please provide a valid call file path.")


if __name__ == "__main__":
    main()
