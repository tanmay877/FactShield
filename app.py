from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import feedparser

app = Flask(__name__)

# ---------------- AI MODEL ----------------
sentiment_analyzer = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ---------------- OFFICIAL RSS FEEDS ----------------
RSS_FEEDS = {
    "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
    "World Health Organization": "https://www.who.int/rss-feeds/news-english.xml",
    "Press Information Bureau": "https://pib.gov.in/rssfeed.aspx",
    "Mint (LiveMint)": "https://www.livemint.com/rss/news",
    "The Indian Express": "https://indianexpress.com/feed/",
    "Aaj Tak": "https://www.aajtak.in/rssfeeds/?id=home",
    "Google News (India)": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
}


# ---------------- HELPER: FETCH HEADLINES ----------------
def fetch_headlines():
    headlines = []
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            headlines.append({
                "source": source,
                "title": entry.title.lower()
            })
    return headlines

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    text = request.json["content"].lower()
    score = 100
    findings = []

    # ---------------- LANGUAGE RISK ----------------
    if "whatsapp" in text or "forwarded" in text:
        score -= 30
        findings.append("Unverified forwarded message")

    if any(w in text for w in ["breaking", "urgent", "panic", "deadly", "shocking"]):
        score -= 25
        findings.append("Alarmist language detected")

    # ---------------- PUBLIC FIGURE SANITY ----------------
    if any(n in text for n in ["modi", "prime minister"]) and "died" in text:
        score = min(score, 15)
        findings.append("Unverified death claim about public figure")

    # ---------------- REAL FACT CHECK (RSS) ----------------
    headlines = fetch_headlines()
    matched_sources = set()

    for item in headlines:
        # simple keyword overlap check
        for word in text.split():
            if len(word) > 4 and word in item["title"]:
                matched_sources.add(item["source"])

    if len(matched_sources) >= 2:
        score += 30
        findings.append(
            f"Claim confirmed by multiple trusted sources: {', '.join(matched_sources)}"
        )
    elif len(matched_sources) == 1:
        score += 10
        findings.append(
            f"Partial confirmation found from {list(matched_sources)[0]}"
        )
    else:
        score -= 25
        findings.append("No confirmation found in official news sources")

    # ---------------- AI EMOTIONAL CHECK ----------------
    ai = sentiment_analyzer(text[:512])[0]
    if ai["label"] == "NEGATIVE" and ai["score"] > 0.85:
        score -= 15
        findings.append("AI detected emotionally manipulative content")

    # ---------------- NORMALIZE ----------------
    score = max(0, min(score, 95))

    if score >= 70:
        status, color = "Likely True", "high"
    elif score >= 40:
        status, color = "Unverified", "medium"
    else:
        status, color = "Likely False", "low"

    return jsonify({
        "score": score,
        "status": status,
        "color": color,
        "findings": findings
    })

if __name__ == "__main__":
    app.run(debug=True)
