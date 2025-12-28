from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import feedparser
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------------- AI MODELS ----------------
sentiment_analyzer = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- RSS FEEDS (TRUSTED SOURCES) ----------------
RSS_FEEDS = {
    "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
    "World Health Organization": "https://www.who.int/rss-feeds/news-english.xml",
    "Press Information Bureau": "https://pib.gov.in/rssfeed.aspx",
    "Mint (LiveMint)": "https://www.livemint.com/rss/news",
    "The Indian Express": "https://indianexpress.com/feed/",
    "Aaj Tak": "https://www.aajtak.in/rssfeeds/?id=home",
    "Google News": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
}

# ---------------- HELPERS ----------------
def is_news_checkable(text):
    keywords = [
        "died", "killed", "announced", "issued", "confirmed",
        "declared", "reported", "arrested", "resigned",
        "alert", "advisory", "launched"
    ]
    return any(k in text for k in keywords)

def extract_core_terms(text):
    stopwords = {
        "the", "is", "to", "of", "and", "a", "will", "in",
        "on", "for", "that", "has", "have", "with"
    }
    return [w for w in text.split() if w not in stopwords and len(w) > 4]

def fetch_recent_headlines():
    headlines = []
    now = datetime.now()

    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
                if now - published > timedelta(days=2):
                    continue

            headlines.append({
                "source": source,
                "title": entry.title.lower()
            })
    return headlines

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    text = request.json["content"].lower()
    score = 100
    findings = []

    # -------- CLAIM TYPE CHECK --------
    if not is_news_checkable(text):
        return jsonify({
            "score": 30,
            "status": "Not Fact-Checkable",
            "color": "medium",
            "findings": [
                "This statement is an opinion, prediction, or non-news claim"
            ]
        })

    # -------- BASIC LANGUAGE RISK --------
    if "whatsapp" in text or "forwarded" in text:
        score -= 30
        findings.append("Unverified forwarded message")

    if any(w in text for w in ["breaking", "urgent", "panic", "shocking", "deadly"]):
        score -= 25
        findings.append("Alarmist language detected")

    # -------- PUBLIC FIGURE SANITY --------
    if any(n in text for n in ["modi", "prime minister"]) and "died" in text:
        score = min(score, 15)
        findings.append("Unverified death claim about public figure")

    # -------- SEMANTIC FACT CHECK --------
    headlines = fetch_recent_headlines()
    matched_sources = set()

    claim_embedding = semantic_model.encode(text, convert_to_tensor=True)
    core_terms = extract_core_terms(text)

    for item in headlines:
        headline_embedding = semantic_model.encode(item["title"], convert_to_tensor=True)
        similarity = util.cos_sim(claim_embedding, headline_embedding).item()

        term_overlap = sum(1 for t in core_terms if t in item["title"])

        if similarity > 0.6 and term_overlap >= 2:
            matched_sources.add(item["source"])

    # -------- CONFIRMATION LOGIC --------
    strong_sources = [s for s in matched_sources if s != "Google News"]

    if len(strong_sources) >= 2:
        score += 35
        findings.append(
            f"Confirmed by multiple trusted sources: {', '.join(strong_sources)}"
        )
    elif len(strong_sources) == 1:
        score += 10
        findings.append(
            f"Partial confirmation from {strong_sources[0]}"
        )
    else:
        score -= 30
        findings.append("No reliable confirmation found in trusted news sources")

    # -------- AI MANIPULATION CHECK --------
    ai = sentiment_analyzer(text[:512])[0]
    if ai["label"] == "NEGATIVE" and ai["score"] > 0.85:
        score -= 15
        findings.append("Emotionally manipulative language detected")

    # -------- NORMALIZE --------
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
