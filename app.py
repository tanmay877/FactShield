from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import datetime

app = Flask(__name__)
CORS(app)

# Trusted sources whitelist
TRUSTED_SOURCES = [
    "bbc", "who", "pib", "indianexpress", "livemint",
    "aajtak", "ndtv", "reuters", "timesofindia"
]

# Alarmist / fake indicators
ALARM_WORDS = [
    "breaking", "urgent", "shocking", "deadly",
    "share immediately", "forwarded", "whatsapp",
    "secret", "hidden truth"
]

# Opinion indicators
OPINION_WORDS = [
    "will happen", "might", "could", "prediction",
    "expected", "possible"
]

@app.route("/")
def home():
    return "FactShield Backend is running"

@app.route("/check", methods=["POST"])
def check():
    data = request.json
    text = data.get("content", "").lower()

    score = 100
    findings = []

    # ---------- SOURCE CHECK ----------
    trusted = any(src in text for src in TRUSTED_SOURCES)
    if trusted:
        findings.append("Verified by trusted news sources")
    else:
        score -= 25
        findings.append("No trusted source detected")

    # ---------- ALARMIST LANGUAGE ----------
    if any(word in text for word in ALARM_WORDS):
        score -= 30
        findings.append("Alarmist / emotionally manipulative language")

    # ---------- OPINION / PREDICTION ----------
    if any(word in text for word in OPINION_WORDS):
        score -= 15
        findings.append("Opinion or prediction, not a verifiable fact")

    # ---------- DATE CHECK ----------
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if year_match:
        year = int(year_match.group())
        current_year = datetime.datetime.now().year
        if year < current_year - 1:
            score -= 15
            findings.append("Information may be outdated")

    # ---------- EXTREME CLAIM CHECK ----------
    if "died" in text or "dead" in text:
        score -= 30
        findings.append("Extraordinary claim requires strong evidence")

    # Clamp score
    score = max(0, min(100, score))

    # ---------- FINAL STATUS ----------
    if score >= 75:
        status = "High Credibility"
        color = "green"
    elif score >= 45:
        status = "Medium Credibility"
        color = "yellow"
    else:
        status = "Low Credibility"
        color = "red"

    return jsonify({
        "score": score,
        "status": status,
        "color": color,
        "findings": findings if findings else ["No major credibility issues detected"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
