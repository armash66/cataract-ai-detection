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

    let stream = null;
    let cameraActive = false;
    let imageFromCamera = false;
    let showingGradcam = false;

    const originalSrc = image ? image.getAttribute("src") : null;
    const gradcamSrc = gradToggleBtn ? gradToggleBtn.dataset.gradcam : null;

    /* =====================================================
       🔹 ADDITION 1: RESULT → data-result AUTO WIRING
       (activates red/green accent automatically)
    ===================================================== */
    if (resultPanel && resultName) {
        const text = resultName.innerText.toLowerCase();
        if (text.includes("normal")) {
            resultPanel.dataset.result = "normal";
        } else if (text.includes("cataract")) {
            resultPanel.dataset.result = "cataract";
        }
    }

    /* =====================================================
       INTERNAL CAMERA UTILITIES (UNCHANGED)
    ===================================================== */
    function stopCameraStream() {
        if (stream) {
            stream.getTracks().forEach(track => {
                try { track.stop(); } catch (_) {}
            });
            stream = null;
        }

        if (video) video.srcObject = null;
        cameraActive = false;
        document.body.classList.remove("camera-live");
        unlockScan();
    }

    /* =====================================================
       🔹 ADDITION 2: LOCK / UNLOCK SCAN IMAGE
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

    /* ===============================
       CAMERA DROPDOWN SLIDE (UNCHANGED)
    ================================ */

    if (cameraToggleBtn && cameraDropdown) {
        cameraDropdown.style.maxHeight = "0px";
        cameraDropdown.style.overflow = "hidden";
        cameraDropdown.style.transition = "max-height 0.35s ease, opacity 0.25s ease";
        cameraDropdown.style.opacity = "0";

        cameraToggleBtn.addEventListener("click", () => {
            const open = cameraDropdown.classList.contains("open");

            if (open) {
                cameraDropdown.style.maxHeight = "0px";
                cameraDropdown.style.opacity = "0";
                cameraDropdown.classList.remove("open");
            } else {
                cameraDropdown.style.maxHeight = "320px";
                cameraDropdown.style.opacity = "1";
                cameraDropdown.classList.add("open");
            }
        });
    }

    /* ===============================
       ACK GATE (UNCHANGED)
    ================================ */

    if (ackCheckbox && startCameraBtn) {
        ackCheckbox.onchange = () => {
            startCameraBtn.disabled = !ackCheckbox.checked;
        };
    }

    /* ===============================
       START CAMERA (MINOR ADDITION)
    ================================ */

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

                /* 🔹 ADDITION: constrain live camera height */
                video.style.maxHeight = "260px";

                cameraActive = true;
                document.body.classList.add("camera-live");
                cameraSection?.classList.remove("camera-hidden");

                lockScan();

            } catch (err) {
                alert("Camera access denied or unavailable.");
                stopCameraStream();
            }
        };
    }

    /* ===============================
       STOP CAMERA (UNCHANGED + SAFE)
    ================================ */

    const stopBtn = document.createElement("button");
    stopBtn.id = "stopCameraBtn";
    stopBtn.innerText = "Stop Camera";

    if (captureBtn && captureBtn.parentElement) {
        const controls = document.createElement("div");
        controls.className = "camera-controls";
        captureBtn.parentElement.insertBefore(controls, captureBtn);
        controls.appendChild(captureBtn);
        controls.appendChild(stopBtn);
    }

    stopBtn.onclick = () => {
        fadeOutCamera();
        stopCameraStream();
    };

    function fadeOutCamera() {
        if (!cameraSection) return;
        cameraSection.style.opacity = "0";
        setTimeout(() => {
            cameraSection.style.opacity = "";
        }, 250);
    }

    /* ===============================
       CAPTURE FRAME (AUTO-STOP)
    ================================ */

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
            });

            fadeOutCamera();
            stopCameraStream();
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

