/* =====================================================
   EyeGPT Unified UI Logic
   ===================================================== */

document.addEventListener("DOMContentLoaded", () => {

    const cameraToggleBtn = document.getElementById("cameraToggleBtn");
    const cameraDropdown = document.getElementById("cameraDropdown");
    const cameraSection = document.getElementById("cameraSection");
    const ackCheckbox = document.getElementById("ackCheckbox");
    const startCameraBtn = document.getElementById("startCameraBtn");
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const captureBtn = document.getElementById("captureBtn");\n    const stopCameraBtn = document.getElementById("stopCameraBtn");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const submitPreviewBtn = document.getElementById("submitPreviewBtn");
    const fileInput = document.getElementById("fileInput");
    const uploadForm = document.getElementById("uploadForm");

    const image = document.getElementById("mainImage");
    const gradToggleBtn = document.getElementById("gradToggleBtn");
    const overlaySlider = document.getElementById("overlaySlider");
    const gradOverlay = document.getElementById("gradOverlay");

    let originalSrc = image ? image.src : "";
    let gradcamSrc = gradToggleBtn ? gradToggleBtn.dataset.gradcam : "";
    let overlayVisible = true;

    const scanBtn = uploadForm?.querySelector("button[type='submit']");
    const resultPanel = document.querySelector(".result-panel");
    const resultName = document.querySelector(".result-name");
    const xaiExplain = document.getElementById("xaiExplain");

    if (xaiExplain && resultName) {
        const resultText = resultName.innerText.toLowerCase();

        if (resultText.includes("cataract")) {
            xaiExplain.innerHTML = `
                <div class="panel-header">
                    <h2>Explainability Report</h2>
                    <span class="panel-sub">Grad-CAM</span>
                </div>
                <ul class="explain-list">
                    <li>Highlighted regions influenced the cataract prediction.</li>
                    <li>Focus areas may include lens opacity patterns.</li>
                    <li>This visualization is for transparency, not diagnosis.</li>
                </ul>
            `;
        } else if (resultText.includes("normal")) {
            xaiExplain.innerHTML = `
                <div class="panel-header">
                    <h2>Explainability Report</h2>
                    <span class="panel-sub">Grad-CAM</span>
                </div>
                <ul class="explain-list">
                    <li>The model found no strong cataract-related patterns.</li>
                    <li>Highlighted regions show areas checked and ruled out.</li>
                    <li>This does not guarantee absence of disease.</li>
                </ul>
            `;
        }
    }

    let stream = null;
    let cameraActive = false;
    let imageFromCamera = false;

    function stopCameraStream() {
        if (stream) {
            stream.getTracks().forEach(track => {
                try { track.stop(); } catch (_) {}
            });
            stream = null;
        }

        if (video) {
            video.srcObject = null;
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

    if (ackCheckbox && startCameraBtn) {
        ackCheckbox.onchange = () => {
            startCameraBtn.disabled = !ackCheckbox.checked;
        };
    }

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

    if (stopCameraBtn) {
        stopCameraBtn.onclick = () => {
            if (!cameraActive) return;
            stopCameraStream();
        };
    }

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
                    previewImage.style.display = "block";
                }

                if (previewSection) {
                    previewSection.style.display = "block";
                }

                imageFromCamera = true;
                disableGradCam();

                stopCameraStream();
            });
        };
    }

    if (submitPreviewBtn && uploadForm) {
        submitPreviewBtn.onclick = () => uploadForm.submit();
    }

    if (overlaySlider && gradOverlay) {
        overlaySlider.addEventListener("input", () => {
            gradOverlay.style.opacity = overlaySlider.value / 100;
        });
    }

    if (gradToggleBtn && gradOverlay && gradcamSrc) {
        gradToggleBtn.addEventListener("click", () => {
            if (imageFromCamera) return;
            overlayVisible = !overlayVisible;
            gradOverlay.style.display = overlayVisible ? "block" : "none";
            gradToggleBtn.innerText = overlayVisible ? "Overlay Attention" : "Show Overlay";
        });
    } else if (gradToggleBtn) {
        gradToggleBtn.disabled = true;
        gradToggleBtn.innerText = "Overlay unavailable";
    }

    function disableGradCam() {
        if (!gradToggleBtn) return;
        gradToggleBtn.disabled = true;
        gradToggleBtn.innerText = "Overlay unavailable for camera images";

        if (gradOverlay) {
            gradOverlay.style.display = "none";
        }
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            imageFromCamera = false;
            originalSrc = image ? image.src : originalSrc;
            if (gradToggleBtn && gradcamSrc) {
                gradToggleBtn.disabled = false;
                gradToggleBtn.innerText = "Overlay Attention";
            }
            if (gradOverlay) {
                gradOverlay.style.display = "block";
            }
        });
    }

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

    if (uploadForm && resultPanel) {
        uploadForm.addEventListener("submit", () => {
            lockScan();

            const placeholder = document.createElement("div");
            placeholder.id = "loadingPlaceholder";
            placeholder.innerText = "Analyzing image...";
            placeholder.style.textAlign = "center";
            placeholder.style.padding = "24px";
            placeholder.style.opacity = "0.6";

            resultPanel.appendChild(placeholder);
        });
    }

    window.addEventListener("beforeunload", stopCameraStream);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) stopCameraStream();
    });

});

