let currentScore = 0;

function checkCredibility() {
    const text = document.getElementById("news").value;

    fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text })
    })
    .then(res => res.json())
    .then(data => {
        currentScore = data.score;

        document.getElementById("result").style.display = "flex";
        document.getElementById("scoreValue").innerText = data.score;
        document.getElementById("statusText").innerText = data.status;
        document.getElementById("statusText").className = "status " + data.color;

        const gauge = document.getElementById("gaugeFill");
        gauge.style.strokeDashoffset = 440 - (data.score / 100) * 440;

        if (data.score >= 70) gauge.style.stroke = "#4ade80";
        else if (data.score >= 40) gauge.style.stroke = "#facc15";
        else gauge.style.stroke = "#f87171";

        const list = document.getElementById("findingsList");
        list.innerHTML = "";
        data.findings.forEach(f => list.innerHTML += `<li>⚠️ ${f}</li>`);
    });
}

function shareAnyway() {
    const text = document.getElementById("news").value;

    if (currentScore >= 70) {
        navigator.clipboard.writeText(text);
        alert("✅ Content copied. This appears credible.");

    } else if (currentScore >= 40) {
        if (confirm("⚠️ Medium credibility. Share responsibly?")) {
            navigator.clipboard.writeText(text);
        }

    } else {
        if (confirm("🚨 LOW credibility! Sharing may cause harm. Continue?")) {
            navigator.clipboard.writeText(text);
        }
    }
}
