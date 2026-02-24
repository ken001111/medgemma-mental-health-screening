"""
iPhone-style demo GUI for the mental health screening application.
Run with: python gui_app.py
Then open http://127.0.0.1:5001 in a browser (use a narrow window or mobile view for iPhone look).
"""
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
import re
from datetime import datetime
from database import ScreeningDatabase
from config import DATA_DIR, DB_PATH, GEMINI_API_KEY, GEMINI_MODEL

app = Flask(__name__, static_folder="gui_static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "data", "gui_uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy init screening app only when processing (heavy)
_screening_app = None
_db = None


def get_db():
    global _db
    if _db is None:
        _db = ScreeningDatabase()
    return _db


def get_screening_app():
    global _screening_app
    if _screening_app is None:
        from main_app import MentalHealthScreeningApp
        _screening_app = MentalHealthScreeningApp()
    return _screening_app


@app.route("/")
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "gui_static"), "index.html")


@app.route("/api/recordings")
def api_recordings():
    """List recordings for the GUI, grouped by day."""
    try:
        db = get_db()
        soldier_id = request.args.get("soldier_id") or None
        recordings = db.get_recordings_for_gui(soldier_id=soldier_id)
        by_day = {}
        for r in recordings:
            ts = r.get("call_timestamp")
            if ts is None:
                ts = r.get("created_at") or datetime.now()
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    try:
                        ts = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts = datetime.now()
            date_key = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M")
            if date_key not in by_day:
                by_day[date_key] = []
            r["date"] = date_key
            r["time"] = time_str
            by_day[date_key].append(r)
        days = sorted(by_day.keys(), reverse=True)
        return jsonify({"days": days, "by_day": by_day, "recordings": recordings}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/process_recording", methods=["POST"])
def api_process_recording():
    """Upload audio file and run the screening pipeline."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        soldier_id = request.form.get("soldier_id", "DEMO_USER")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filename = secure_filename(file.filename) or "recording.webm"
        if not filename.lower().endswith((".webm", ".mp3", ".wav", ".m4a", ".ogg")):
            filename += ".webm"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        app_instance = get_screening_app()
        result = app_instance.process_call(
            call_path=filepath,
            soldier_id=soldier_id,
            call_timestamp=datetime.now(),
        )
        return jsonify({
            "success": True,
            "call_id": result["call_id"],
            "soldier_id": result["soldier_id"],
            "scores": result.get("scores", {}),
            "severity": result.get("risk_assessment", {}).get("severity", "low"),
            "report_path": result.get("report_path"),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/send_to_clinician/<call_id>", methods=["POST"])
def api_send_to_clinician(call_id):
    """Mark report as sent to clinician (after consent)."""
    try:
        get_db().mark_sent_to_clinician(call_id)
        return jsonify({"success": True, "call_id": call_id, "message": "Report sent to clinician for deeper review."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _chat_with_gemini(messages: list, context: dict) -> str:
    """Call Google Gemini with conversation history and screening context."""
    if not GEMINI_API_KEY:
        return "Chat is not configured. Set the GEMINI_API_KEY environment variable to enable chatting with the Google AI."
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    severity = context.get("severity", "unknown")
    phq9 = context.get("phq9_score")
    report_snippet = (context.get("report_content") or "")[:2000]
    system_prompt = (
        "You are a supportive, respectful assistant for soldiers discussing their mental health screening results. "
        "You are not a doctor or clinician. Do not give medical advice or diagnoses. "
        "Encourage professional care when appropriate. Be brief and kind. "
        "Context from this soldier's screening: severity=" + str(severity)
    )
    if phq9 is not None:
        system_prompt += f", PHQ-9 score={phq9}"
    if report_snippet:
        system_prompt += ". Report summary (for context only): " + report_snippet[:1200]
    system_prompt += "."
    # Try configured model first, then fallbacks (model names change over time).
    # Prefer the exact IDs returned by genai.list_models() (often prefixed with "models/").
    model_ids = [
        GEMINI_MODEL,
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-3-flash-preview",
        "models/gemini-3.1-pro-preview",
    ]
    model_ids = [m for m in model_ids if m]  # dedupe
    seen = set()
    model_ids = [m for m in model_ids if m not in seen and not seen.add(m)]
    QUOTA_MSG = (
        "Google AI (Gemini) quota or rate limit was exceeded. "
        "Check your plan and billing at https://ai.google.dev/gemini-api/docs/rate-limits and "
        "monitor usage at https://ai.dev/rate-limit. If you're on the free tier, you may have hit the daily or per-minute limit—try again later or enable billing."
    )

    last_error = None
    def _normalize_model_id(mid: str) -> str:
        mid = (mid or "").strip()
        if not mid:
            return ""
        if mid.startswith("models/"):
            return mid
        # Some examples in docs omit the prefix; the API uses "models/<id>".
        return "models/" + mid

    for model_id in [_normalize_model_id(m) for m in model_ids if m]:
        for attempt in range(2):  # one normal try, one retry after delay for 429
            try:
                model = genai.GenerativeModel(model_id, system_instruction=system_prompt)
                history = []
                for m in messages[:-1]:
                    role = "user" if m.get("role") == "user" else "model"
                    history.append({"role": role, "parts": [m.get("content", "")]})
                chat = model.start_chat(history=history)
                last_content = (messages[-1].get("content", "") if messages else "") or "Hello"
                response = chat.send_message(last_content)
                return response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    break  # try next model, no retry
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "rate-limits" in err_str:
                    if attempt == 0:
                        # Optional: wait and retry once (e.g. per-minute limit)
                        wait_s = 2.0
                        match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_str, re.I)
                        if match:
                            try:
                                wait_s = min(float(match.group(1)) + 0.5, 10.0)
                            except Exception:
                                pass
                        time.sleep(wait_s)
                        continue
                    return QUOTA_MSG
                return "Sorry, I couldn't get a response from the AI. " + str(e)
    return "Sorry, I couldn't get a response from the AI. " + (
        str(last_error)
        if last_error
        else "No supported model found. Try setting GEMINI_MODEL to a value from list_models() (e.g. models/gemini-flash-latest)."
    )


def _chat_with_medgemma_local(messages: list, context: dict) -> str:
    """
    Local fallback chat using the already-downloaded MedGemma model (Google) via HuggingFace.
    This avoids external API quotas and still satisfies 'Google-based model' for the demo.
    """
    try:
        app_instance = get_screening_app()
        mg = getattr(getattr(app_instance, "analyzer", None), "medgemma", None)
        if not mg or not hasattr(mg, "is_available") or not mg.is_available():
            return ""

        # Build a concise chat prompt with context
        severity = context.get("severity", "unknown")
        phq9 = context.get("phq9_score")
        report = (context.get("report_content") or "")[:1200]
        transcript = (context.get("transcript") or "")[:1200]

        prompt = (
            "You are a supportive assistant for soldiers discussing their mental health screening results. "
            "You are not a clinician. Do not provide medical advice or diagnoses. "
            "Encourage seeking professional care when appropriate.\n\n"
            f"SCREENING CONTEXT:\n- Severity: {severity}\n"
        )
        if phq9 is not None:
            prompt += f"- PHQ-9: {phq9}\n"
        if report:
            prompt += f"- Report summary: {report}\n"
        if transcript:
            prompt += f"- Transcript excerpt: {transcript}\n"

        prompt += "\nCONVERSATION:\n"
        for m in messages[-8:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            content = (m.get("content") or "").strip()
            if content:
                prompt += f"{role}: {content}\n"
        prompt += "Assistant:"

        inputs = mg.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(mg.device)
        outputs = mg.model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=mg.tokenizer.eos_token_id,
        )
        text = mg.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Basic stop if the model starts another turn marker
        for stop in ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:"]:
            if stop in text:
                text = text.split(stop, 1)[0].strip()
        return text
    except Exception:
        return ""


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with Google Gemini about the screening (context from call_id)."""
    try:
        data = request.get_json() or {}
        messages = data.get("messages", [])
        call_id = data.get("call_id")
        if not messages:
            return jsonify({"error": "messages required"}), 400
        context = {}
        if call_id:
            ctx = get_db().get_call_context_for_chat(call_id)
            if ctx:
                context = {
                    "severity": ctx.get("severity", "unknown"),
                    "phq9_score": ctx.get("phq9_score"),
                    "report_content": ctx.get("report_content"),
                    "transcript": ctx.get("transcript"),
                }
        reply = _chat_with_gemini(messages, context)
        # If Gemini is blocked by quota or model availability, fall back to local MedGemma (Google).
        if (
            "quota or rate limit was exceeded" in (reply or "").lower()
            or "no supported model found" in (reply or "").lower()
            or "not found" in (reply or "").lower()
            or "not supported" in (reply or "").lower()
        ):
            local_reply = _chat_with_medgemma_local(messages, context)
            if local_reply:
                reply = "Using local MedGemma (Google) because Gemini API is unavailable.\n\n" + local_reply
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Demo GUI: open http://127.0.0.1:5001 in a browser (use narrow window or mobile view for iPhone look).")
    app.run(host="127.0.0.1", port=5001, debug=False)
