/* =====================================================
   EyeGPT Unified UI Logic
   - Image toggle (Original <-> Grad-CAM)
   - Overlay opacity control
   - Confidence ring animation
   - Camera lifecycle (merged from camera.js)
   ===================================================== */

document.addEventListener("DOMContentLoaded", () => {

    /* ===============================
       DOM REFERENCES
    ================================ */

    const cameraToggleBtn = document.getElementById("cameraToggleBtn");
    const cameraDropdown = document.getElementById("cameraDropdown");
    const cameraSection = document.getElementById("cameraSection");
    const ackCheckbox = document.getElementById("ackCheckbox");
    const startCameraBtn = document.getElementById("startCameraBtn");
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const captureBtn = document.getElementById("captureBtn");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const submitPreviewBtn = document.getElementById("submitPreviewBtn");
    const fileInput = document.getElementById("fileInput");
    const uploadForm = document.getElementById("uploadForm");

    const image = document.getElementById("mainImage");
    const gradToggleBtn = document.getElementById("gradToggleBtn");
    const overlaySlider = document.getElementById("overlaySlider");

    const scanBtn = uploadForm?.querySelector("button[type='submit']");
    const resultPanel = document.querySelector(".result-panel");
    const resultName = document.querySelector(".result-name");
    const xaiExplain = document.getElementById("xaiExplain");

    if (xaiExplain && resultName) {
        const resultText = resultName.innerText.toLowerCase();

        if (resultText.includes("cataract")) {
            xaiExplain.innerHTML = `
                <strong>Visual explanation (XAI)</strong>
                <ul>
                    <li>Highlighted regions influenced the model’s cataract prediction</li>
                    <li>Focus areas may include lens opacity patterns</li>
                    <li>This visualization is for transparency, not diagnosis</li>
                </ul>
            `;
        } else if (resultText.includes("normal")) {
            xaiExplain.innerHTML = `
                <strong>Visual explanation (XAI)</strong>
                <ul>
                    <li>The model found no strong cataract-related patterns</li>
                    <li>Highlighted regions show areas checked and ruled out</li>
                    <li>This does not guarantee absence of disease</li>
                </ul>
            `;
        }
    }

  /* =====================================================
   CAMERA STATE
===================================================== */

let stream = null;
let cameraActive = false;
let imageFromCamera = false;

/* =====================================================
   CAMERA UTILITIES
===================================================== */

function stopCameraStream() {
    if (stream) {
        stream.getTracks().forEach(track => {
            try { track.stop(); } catch (_) {}
        });
        stream = null;
    }

    if (video) {
        video.srcObject = null; // black screen
    }

    resetCameraUI();
}

function resetCameraUI() {
    cameraActive = false;

    document.body.classList.remove("camera-live");

    if (cameraSection) {
        cameraSection.classList.add("camera-hidden");
    }

    unlockScan();
}

/* =====================================================
   LOCK / UNLOCK SCAN
===================================================== */

function lockScan() {
    if (!scanBtn) return;
    scanBtn.disabled = true;
    scanBtn.style.opacity = "0.5";
    scanBtn.style.pointerEvents = "none";
}

function unlockScan() {
    if (!scanBtn) return;
    scanBtn.disabled = false;
    scanBtn.style.opacity = "1";
    scanBtn.style.pointerEvents = "auto";
}

/* =====================================================
   CAMERA DROPDOWN (SAFE)
===================================================== */

if (cameraToggleBtn && cameraDropdown) {
    cameraDropdown.style.maxHeight = "0px";
    cameraDropdown.style.opacity = "0";
    cameraDropdown.style.overflow = "hidden";
    cameraDropdown.style.transition = "max-height 0.35s ease, opacity 0.25s ease";

    cameraToggleBtn.onclick = () => {
        const open = cameraDropdown.classList.contains("open");

        if (open) {
            cameraDropdown.style.maxHeight = "0px";
            cameraDropdown.style.opacity = "0";
            cameraDropdown.classList.remove("open");

            if (cameraActive) {
                stopCameraStream();
            }
        } else {
            cameraDropdown.style.maxHeight = "320px";
            cameraDropdown.style.opacity = "1";
            cameraDropdown.classList.add("open");
        }
    };
}

/* =====================================================
   ACK GATE
===================================================== */

if (ackCheckbox && startCameraBtn) {
    ackCheckbox.onchange = () => {
        startCameraBtn.disabled = !ackCheckbox.checked;
    };
}

/* =====================================================
   START CAMERA
===================================================== */

if (startCameraBtn && video) {
    startCameraBtn.onclick = async () => {
        if (cameraActive) return;

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: 640, height: 480 },
                audio: false
            });

            video.srcObject = stream;
            video.muted = true;
            video.setAttribute("playsinline", "");
            video.style.objectFit = "cover";

            cameraActive = true;
            document.body.classList.add("camera-live");

            if (cameraSection) {
                cameraSection.classList.remove("camera-hidden");
            }

            lockScan();

        } catch (err) {
            alert("Camera access denied or unavailable.");
            stopCameraStream();
        }
    };
}

/* =====================================================
   STOP CAMERA (NO CAPTURE)
===================================================== */

if (stopCameraBtn) {
    stopCameraBtn.onclick = () => {
        if (!cameraActive) return;
        stopCameraStream();
    };
}

/* =====================================================
   CAPTURE FRAME (ONE HANDLER ONLY)
===================================================== */

if (captureBtn && canvas && video) {
    captureBtn.onclick = () => {
        if (!cameraActive || !video.videoWidth) {
            alert("Camera not ready.");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);

        canvas.toBlob(blob => {
            if (!blob) return;

            const file = new File([blob], "camera.png", { type: "image/png" });
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;

            if (previewImage) {
                previewImage.src = URL.createObjectURL(blob);
            }

            if (previewSection) {
                previewSection.style.display = "block";
            }

            imageFromCamera = true;
            disableGradCam();

            stopCameraStream(); // stop AFTER capture
        });
    };
}


    /* ===============================
       SUBMIT PREVIEW (UNCHANGED)
    ================================ */

    if (submitPreviewBtn && uploadForm) {
        submitPreviewBtn.onclick = () => uploadForm.submit();
    }

    /* ===============================
       IMAGE TOGGLE + OPACITY
    ================================ */

    if (overlaySlider && image) {
        overlaySlider.addEventListener("input", () => {
            image.style.opacity = overlaySlider.value / 100;
        });
    }

    if (gradToggleBtn && image && gradcamSrc) {
        gradToggleBtn.addEventListener("click", () => {
            if (imageFromCamera) return;
            showingGradcam = !showingGradcam;
            image.src = showingGradcam ? gradcamSrc : originalSrc;
        });
    }

    function disableGradCam() {
        if (!gradToggleBtn) return;
        gradToggleBtn.disabled = true;
        gradToggleBtn.innerText = "Visual explanation unavailable for camera images";
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            imageFromCamera = false;
            if (gradToggleBtn) {
                gradToggleBtn.disabled = false;
                gradToggleBtn.innerText = "Toggle visual explanation (Grad-CAM)";
            }
        });
    }

    /* ===============================
       CONFIDENCE RING (UNCHANGED)
    ================================ */

    const confidenceCircle = document.getElementById("confidenceCircle");
    if (confidenceCircle?.dataset.value) {
        const value = Number(confidenceCircle.dataset.value);
        const circumference = 314;
        const offset = circumference - (value / 100) * circumference;

        confidenceCircle.style.strokeDasharray = circumference;
        confidenceCircle.style.strokeDashoffset = circumference;

        requestAnimationFrame(() => {
            confidenceCircle.style.transition = "stroke-dashoffset 1.2s ease";
            confidenceCircle.style.strokeDashoffset = offset;
        });
    }

    /* =====================================================
       🔹 ADDITION 3: SKELETON LOADING ON INFERENCE
    ===================================================== */
    if (uploadForm && resultPanel) {
        uploadForm.addEventListener("submit", () => {

            lockScan();

            if (image) image.style.display = "none";

            const placeholder = document.createElement("div");
            placeholder.id = "loadingPlaceholder";
            placeholder.innerText = "Analyzing image…";
            placeholder.style.textAlign = "center";
            placeholder.style.padding = "40px";
            placeholder.style.opacity = "0.6";

            resultPanel.appendChild(placeholder);
        });
    }

    /* ===============================
       SAFETY CLEANUP
    ================================ */

    window.addEventListener("beforeunload", stopCameraStream);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) stopCameraStream();
    });

});

const historyBtn = document.getElementById("historyBtn");

if (historyBtn) {
    historyBtn.addEventListener("click", () => {
        window.location.href = "/history";
    });
}
