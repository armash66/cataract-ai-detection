let originalSrc = "";
let gradcamSrc = "";
let showingGradcam = false;

document.querySelectorAll(".history-card").forEach(card => {
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
let currentIndex = -1;
const cards = Array.from(document.querySelectorAll(".history-card"));

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

const confidences = [...document.querySelectorAll(".history-card")]
  .map(c => Number(c.dataset.confidence));

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
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.fillRect(i * 28, 110 - barHeight, 20, barHeight);
    });
}

let pinned = [];

document.querySelectorAll(".history-card").forEach(card => {
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

document.querySelectorAll(".history-card").forEach(card => {
    card.addEventListener("mouseenter", () => {
        document.getElementById("detailImage").src = card.dataset.image;
    });
});
