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
