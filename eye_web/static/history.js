let originalSrc = "";
let gradcamSrc = "";
let showingGradcam = false;

const cards = Array.from(document.querySelectorAll(".history-card"));
const detailSeverity = document.getElementById("detailSeverity");
const copyDetailBtn = document.getElementById("copyDetailBtn");

cards.forEach(card => {
    card.addEventListener("click", () => selectScan(card));
});

function selectScan(card) {
    const img = document.getElementById("detailImage");
    const placeholder = document.getElementById("detailPlaceholder");

    img.onload = () => {
        img.style.display = "block";
        placeholder.style.display = "none";
    };

    originalSrc = card.dataset.image;
    gradcamSrc = card.dataset.gradcam;
    showingGradcam = false;

    document.getElementById("detailImage").src = originalSrc;
    document.getElementById("detailPrediction").innerText = card.dataset.prediction;
    document.getElementById("detailConfidence").innerText =
        card.dataset.confidence + "%";
    document.getElementById("detailTime").innerText = card.dataset.time;

    if (detailSeverity) {
        const value = Number(card.dataset.confidence);
        detailSeverity.textContent = value >= 80 ? "Low" : value >= 60 ? "Medium" : "High";
    }

    document.getElementById("downloadGradcam").href = gradcamSrc;
    document.getElementById("deleteForm").action =
        `/delete/${card.dataset.id}`;

    document.getElementById("toggleXaiBtn").innerText =
        "Show visual explanation (Grad-CAM)";
}

document.getElementById("toggleXaiBtn").addEventListener("click", () => {
    const img = document.getElementById("detailImage");

    showingGradcam = !showingGradcam;
    img.src = showingGradcam ? gradcamSrc : originalSrc;

    document.getElementById("toggleXaiBtn").innerText =
        showingGradcam
            ? "Show original image"
            : "Show visual explanation (Grad-CAM)";
});

if (copyDetailBtn) {
    copyDetailBtn.addEventListener("click", async () => {
        const prediction = document.getElementById("detailPrediction").innerText;
        const confidence = document.getElementById("detailConfidence").innerText;
        const time = document.getElementById("detailTime").innerText;
        const text = `CataractGPT Result: ${prediction} (${confidence}) at ${time}`;
        try {
            await navigator.clipboard.writeText(text);
            copyDetailBtn.textContent = "Copied";
            setTimeout(() => copyDetailBtn.textContent = "Copy summary", 1200);
        } catch (_) {
            copyDetailBtn.textContent = "Copy failed";
        }
    });
}

let currentIndex = -1;

document.addEventListener("keydown", (e) => {
    if (!cards.length) return;

    if (e.key === "ArrowDown") {
        currentIndex = Math.min(currentIndex + 1, cards.length - 1);
        cards[currentIndex].click();
        cards[currentIndex].scrollIntoView({ block: "nearest" });
    }

    if (e.key === "ArrowUp") {
        currentIndex = Math.max(currentIndex - 1, 0);
        cards[currentIndex].click();
        cards[currentIndex].scrollIntoView({ block: "nearest" });
    }
});

const confidences = cards.map(c => Number(c.dataset.confidence));

if (confidences.length) {
    const bins = Array(10).fill(0);
    confidences.forEach(v => bins[Math.min(9, Math.floor(v / 10))]++);

    const canvas = document.getElementById("confidenceChart");
    const ctx = canvas.getContext("2d");

    canvas.width = 300;
    canvas.height = 120;

    const max = Math.max(...bins, 1);

    bins.forEach((val, i) => {
        const barHeight = (val / max) * 100;
        ctx.fillStyle = "rgba(56, 189, 248, 0.35)";
        ctx.fillRect(i * 28, 110 - barHeight, 20, barHeight);
    });
}

let pinned = [];

cards.forEach(card => {
    card.addEventListener("contextmenu", e => {
        e.preventDefault();

        if (pinned.length < 2) {
            pinned.push(card.dataset.image);
        }

        if (pinned.length === 2) {
            document.querySelector(".comparison-row").style.display = "block";
            document.getElementById("compareImage").src = pinned[1];
        }
    });
});

cards.forEach(card => {
    card.addEventListener("mouseenter", () => {
        document.getElementById("detailImage").src = card.dataset.image;
    });
});

const demoModeBtn = document.getElementById("demoModeBtn");
const compactModeBtn = document.getElementById("compactModeBtn");
const resetSessionBtn = document.getElementById("resetSessionBtn");

if (demoModeBtn) {
    demoModeBtn.addEventListener("click", () => {
        document.body.classList.toggle("demo-mode");
    });
}

if (compactModeBtn) {
    compactModeBtn.addEventListener("click", () => {
        document.body.classList.toggle("compact-mode");
    });
}

if (resetSessionBtn) {
    resetSessionBtn.addEventListener("click", () => {
        const form = document.querySelector("form[action*='history/clear']");
        if (form) form.submit();
    });
}

const chatBody = document.getElementById("chatBody");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");
const chatSuggestions = document.getElementById("chatSuggestions");

const sendChat = (text) => {
    if (!text) return;
    const userMsg = document.createElement("div");
    userMsg.className = "chat-message user";
    userMsg.textContent = text;
    chatBody.appendChild(userMsg);

    const botMsg = document.createElement("div");
    botMsg.className = "chat-message bot";
    botMsg.textContent = "Analyzing...";
    chatBody.appendChild(botMsg);

    setTimeout(() => {
        botMsg.textContent = "This is a research-only explanation. The confidence reflects model certainty, not clinical severity.";
    }, 600);

    chatBody.scrollTop = chatBody.scrollHeight;
};

if (chatSend && chatInput) {
    chatSend.addEventListener("click", () => {
        sendChat(chatInput.value.trim());
        chatInput.value = "";
    });
}

if (chatSuggestions) {
    chatSuggestions.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        sendChat(btn.textContent.trim());
    });
}
