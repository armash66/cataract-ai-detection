/* ===============================
   ELEMENT REFERENCES
================================ */
const eyeCore = document.getElementById("eyeCore");
const eyeIris = document.getElementById("eyeIris");
const scanLine = document.getElementById("scanLine");
const systemStatus = document.getElementById("systemStatus");
const chatBox = document.getElementById("aiResponse");
const confidenceCircle = document.getElementById("confidenceCircle");
const gradcamImage = document.getElementById("gradcamImage");
const gradSlider = document.getElementById("gradSlider");

/* 🔹 ADDITION: gradcam toggle + rays */
const gradcamSection = document.getElementById("gradcamSection");
const gradToggleBtn = document.getElementById("gradToggleBtn");
const eyeRays = document.querySelector(".eye-rays");

/* ===============================
   EYE TRACKING (PARALLAX)
================================ */
document.addEventListener("mousemove", (e) => {
    if (!eyeIris || !eyeCore) return;

    const rect = eyeCore.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const dx = (e.clientX - centerX) / 30;
    const dy = (e.clientY - centerY) / 30;

    eyeIris.style.transform = `translate(${dx}px, ${dy}px)`;
});

/* ===============================
   SCAN START EFFECT
================================ */
function startScanEffects() {
    if (systemStatus) systemStatus.innerText = "SCANNING";
    if (eyeCore) eyeCore.classList.add("active");

    /* 🔹 ADDITION: activate rays */
    if (eyeRays) eyeRays.classList.add("active");

    if (!scanLine) return;

    scanLine.style.opacity = 1;
    scanLine.style.transition = "none";
    scanLine.style.top = "-10%";

    requestAnimationFrame(() => {
        scanLine.style.transition = "top 1.8s linear, opacity 0.5s";
        scanLine.style.top = "110%";
    });

    setTimeout(() => {
        scanLine.style.opacity = 0;
    }, 1800);
}

/* ===============================
   SCAN END EFFECT
================================ */
function endScanEffects() {
    if (systemStatus) systemStatus.innerText = "ANALYSIS COMPLETE";
    if (eyeCore) eyeCore.classList.remove("active");

    /* 🔹 ADDITION: deactivate rays */
    if (eyeRays) eyeRays.classList.remove("active");

    if (!window.lastResult) return;

    // Color shift based on result
    if (window.lastResult.prediction.toLowerCase().includes("cataract")) {
        eyeCore.style.boxShadow = "0 0 140px rgba(255,0,0,1)";
        if (confidenceCircle) confidenceCircle.style.stroke = "red";
    } else {
        eyeCore.style.boxShadow = "0 0 140px rgba(0,200,255,1)";
        if (confidenceCircle) confidenceCircle.style.stroke = "#00d9ff";
    }

    animateConfidence(window.lastResult.confidence);
    autoExplain();
}

/* ===============================
   CONFIDENCE RING ANIMATION
================================ */
function animateConfidence(value) {
    if (!confidenceCircle) return;

    const circumference = 283;
    const offset = circumference - (value / 100) * circumference;

    confidenceCircle.style.strokeDashoffset = offset;
}

/* ===============================
   GRADCAM OPACITY SLIDER (FIXED)
================================ */
if (gradSlider && gradcamImage) {
    gradSlider.addEventListener("input", () => {
        gradcamImage.style.opacity = gradSlider.value / 100;
    });
}

/* ===============================
   GRADCAM SHOW / HIDE (ADDITION)
================================ */
if (gradToggleBtn && gradcamSection) {
    gradToggleBtn.addEventListener("click", () => {
        const hidden = gradcamSection.classList.toggle("hidden");
        gradToggleBtn.innerText = hidden
            ? "Show Visual Explanation"
            : "Hide Visual Explanation";
    });
}

/* ===============================
   TYPING EFFECT
================================ */
function typeText(element, text, speed = 20) {
    if (!element) return;

    element.innerHTML = "";
    let index = 0;

    function type() {
        if (index < text.length) {
            element.innerHTML += text[index++];
            element.scrollTop = element.scrollHeight;
            setTimeout(type, speed);
        }
    }
    type();
}

/* ===============================
   AUTO CHAT EXPLANATION
================================ */
function autoExplain() {
    if (!window.lastResult || !chatBox) return;

    const { prediction, confidence } = window.lastResult;

    const explanation = `
Result: ${prediction}
Confidence: ${confidence}%

The model analyzed visible lens opacity, contrast loss, and light scattering patterns.
Higher confidence indicates stronger visual features consistent with the prediction.

This is an AI-assisted screening result, not a medical diagnosis.
    `.trim();

    typeText(chatBox, explanation, 18);
}

/* ===============================
   MANUAL CHAT TRIGGER
================================ */
function askEyeGPT() {
    if (!window.lastResult || !chatBox) {
        if (chatBox) chatBox.innerHTML = "No scan data available.";
        return;
    }
    autoExplain();
}

/* ===============================
   HOOK INTO FORM SUBMISSION
================================ */
const uploadForm = document.getElementById("uploadForm");
if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
        startScanEffects();
        if (systemStatus) systemStatus.innerText = "PROCESSING IMAGE";
    });
}

/* ===============================
   RUN ON PAGE LOAD IF RESULT EXISTS
================================ */
window.addEventListener("load", () => {
    if (window.lastResult) {
        setTimeout(endScanEffects, 800);
    }
});
