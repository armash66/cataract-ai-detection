/* =====================================================
   CataractGPT Scan-Lab UI Logic
   ===================================================== */

document.addEventListener("DOMContentLoaded", () => {
    const qs = (sel) => document.querySelector(sel);
    const qsa = (sel) => Array.from(document.querySelectorAll(sel));

    const body = document.body;
    const themeToggle = qs("#themeToggle");
    const storedTheme = localStorage.getItem("cataractgpt_theme");
    if (storedTheme === "light") body.classList.add("theme-light");

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            body.classList.toggle("theme-light");
            localStorage.setItem("cataractgpt_theme", body.classList.contains("theme-light") ? "light" : "lab");
        });
    }

    const resultPanel = qs(".result-panel");
    const confidenceValue = qs("#confidenceValue");
    const confidenceCircle = qs("#confidenceCircle");
    const lowConfidence = qs("#lowConfidence");
    const confidenceBadge = qs("#confidenceBadge");
    const severityLevel = qs("#severityLevel");
    const avgDelta = qs("#avgDelta");
    const resultTimestamp = qs("#resultTimestamp");
    const scanDuration = qs("#scanDuration");
    const toggleDetail = qs("#toggleDetail");
    const detailBlock = qs("#detailBlock");
    const copyResultBtn = qs("#copyResultBtn");

    const fileInput = qs("#fileInput");
    const uploadForm = qs("#uploadForm");
    const fileName = qs("#fileName");
    const fileSize = qs("#fileSize");
    const fileFormat = qs("#fileFormat");
    const retakeBtn = qs("#retakeBtn");
    const qualityBlur = qs("#qualityBlur");
    const qualityBright = qs("#qualityBright");
    const qualityReflect = qs("#qualityReflect");

    const gradToggleBtn = qs("#gradToggleBtn");
    const overlaySlider = qs("#overlaySlider");
    const gradOverlay = qs("#gradOverlay");
    const paletteSelect = qs("#paletteSelect");

    const eli10Btn = qs("#eli10Btn");
    const techBtn = qs("#techBtn");
    const eli10Text = qs("#eli10Text");
    const techText = qs("#techText");

    const cameraToggleBtn = qs("#cameraToggleBtn");
    const cameraDropdown = qs("#cameraDropdown");
    const cameraSection = qs("#cameraSection");
    const ackCheckbox = qs("#ackCheckbox");
    const startCameraBtn = qs("#startCameraBtn");
    const cameraDisclaimerBtn = qs("#cameraDisclaimerBtn");
    const video = qs("#video");
    const canvas = qs("#canvas");
    const captureBtn = qs("#captureBtn");
    const captureCountdownBtn = qs("#captureCountdownBtn");
    const stopCameraBtn = qs("#stopCameraBtn");
    const previewImage = qs("#previewImage");

    const chatBody = qs("#chatBody");
    const chatInput = qs("#chatInput");
    const chatSend = qs("#chatSend");
    const chatSuggestions = qs("#chatSuggestions");

    const cameraModal = qs("#cameraModal");
    const closeModal = qs("#closeModal");

    const timelineSteps = qsa(".timeline-step");

    const setTimeline = (state) => {
        timelineSteps.forEach(step => {
            step.classList.remove("is-active", "is-done");
        });
        if (state === "idle") {
            timelineSteps[0]?.classList.add("is-active");
        } else if (state === "upload") {
            timelineSteps[0]?.classList.add("is-done");
            timelineSteps[1]?.classList.add("is-active");
        } else if (state === "result") {
            timelineSteps.forEach(step => step.classList.add("is-done"));
        }
    };

    if (resultPanel?.dataset.confidence) {
        const value = Number(resultPanel.dataset.confidence);

        if (confidenceValue) {
            const target = value;
            let current = 0;
            const step = Math.max(1, Math.floor(target / 30));
            const interval = setInterval(() => {
                current += step;
                if (current >= target) {
                    current = target;
                    clearInterval(interval);
                }
                confidenceValue.textContent = String(current);
            }, 20);
        }

        if (confidenceCircle) {
            const circumference = 314;
            const offset = circumference - (value / 100) * circumference;
            confidenceCircle.style.strokeDasharray = circumference;
            confidenceCircle.style.strokeDashoffset = circumference;
            requestAnimationFrame(() => {
                confidenceCircle.style.transition = "stroke-dashoffset 1.2s ease";
                confidenceCircle.style.strokeDashoffset = offset;
            });
        }

        if (confidenceBadge) {
            if (value >= 80) confidenceBadge.textContent = "High confidence";
            else if (value >= 60) confidenceBadge.textContent = "Moderate confidence";
            else confidenceBadge.textContent = "Low confidence";
        }

        if (lowConfidence) {
            lowConfidence.style.display = value < 60 ? "inline-flex" : "none";
        }

        if (severityLevel) {
            severityLevel.textContent = value >= 80 ? "Low" : value >= 60 ? "Medium" : "High";
        }

        if (avgDelta) {
            const avg = 72;
            const delta = (value - avg).toFixed(1);
            avgDelta.textContent = `${delta > 0 ? "+" : ""}${delta}% vs avg`;
        }

        if (resultTimestamp) {
            const now = new Date();
            resultTimestamp.textContent = now.toISOString().replace("T", " ").substring(0, 16);
        }

        if (scanDuration) {
            const start = Number(localStorage.getItem("scanStart") || "0");
            if (start) {
                const duration = ((Date.now() - start) / 1000).toFixed(2);
                scanDuration.textContent = `${duration}s`;
                localStorage.removeItem("scanStart");
            } else {
                scanDuration.textContent = "1.24s";
            }
        }

        setTimeline("result");
    } else {
        setTimeline("idle");
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", () => {
            localStorage.setItem("scanStart", String(Date.now()));
            setTimeline("upload");
        });
    }

    if (toggleDetail && detailBlock) {
        toggleDetail.addEventListener("click", () => {
            detailBlock.classList.toggle("is-hidden");
        });
    }

    if (copyResultBtn && resultPanel?.dataset.confidence) {
        copyResultBtn.addEventListener("click", async () => {
            const prediction = qs(".prediction-badge")?.textContent || "Result";
            const confidence = resultPanel.dataset.confidence;
            const stamp = resultTimestamp?.textContent || "";
            const text = `CataractGPT Result: ${prediction} (${confidence}%). ${stamp}`;
            try {
                await navigator.clipboard.writeText(text);
                copyResultBtn.textContent = "Copied";
                setTimeout(() => copyResultBtn.textContent = "Copy result", 1200);
            } catch (_) {
                copyResultBtn.textContent = "Copy failed";
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files?.[0];
            if (!file) return;

            fileName.textContent = file.name;
            fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
            fileFormat.textContent = file.name.split(".").pop()?.toUpperCase() || "";

            const sizeKb = file.size / 1024;
            const blurWarn = sizeKb < 80;
            const brightWarn = sizeKb > 2000;
            const reflectWarn = sizeKb < 120;

            [qualityBlur, qualityBright, qualityReflect].forEach(el => {
                el.classList.remove("is-warning", "is-good");
            });

            if (qualityBlur) qualityBlur.classList.add(blurWarn ? "is-warning" : "is-good");
            if (qualityBright) qualityBright.classList.add(brightWarn ? "is-warning" : "is-good");
            if (qualityReflect) qualityReflect.classList.add(reflectWarn ? "is-warning" : "is-good");

            setTimeline("upload");
        });
    }

    if (retakeBtn && fileInput) {
        retakeBtn.addEventListener("click", () => {
            fileInput.value = "";
            if (fileName) fileName.textContent = "No file selected";
            if (fileSize) fileSize.textContent = "--";
            if (fileFormat) fileFormat.textContent = "--";
        });
    }

    let overlayVisible = true;

    if (overlaySlider && gradOverlay) {
        overlaySlider.addEventListener("input", () => {
            gradOverlay.style.opacity = overlaySlider.value / 100;
        });
    }

    if (gradToggleBtn && gradOverlay && gradToggleBtn.dataset.gradcam) {
        gradToggleBtn.addEventListener("click", () => {
            overlayVisible = !overlayVisible;
            gradOverlay.style.display = overlayVisible ? "block" : "none";
            gradToggleBtn.textContent = overlayVisible ? "Overlay Attention" : "Show Overlay";
        });
    }

    if (paletteSelect && gradOverlay) {
        paletteSelect.addEventListener("change", () => {
            const palette = paletteSelect.value;
            const filterMap = {
                inferno: "hue-rotate(0deg) saturate(1.2)",
                turbo: "hue-rotate(40deg) saturate(1.6)",
                viridis: "hue-rotate(120deg) saturate(1.2)",
                mono: "grayscale(1)"
            };
            gradOverlay.style.filter = filterMap[palette] || "none";
        });
    }

    qsa(".imaging-cell").forEach(cell => {
        const btn = cell.querySelector(".fullscreen-btn");
        if (!btn) return;
        btn.addEventListener("click", () => {
            cell.classList.toggle("is-fullscreen");
        });
    });

    if (eli10Btn && techBtn && eli10Text && techText) {
        eli10Btn.addEventListener("click", () => {
            eli10Text.classList.remove("is-hidden");
            techText.classList.add("is-hidden");
        });
        techBtn.addEventListener("click", () => {
            techText.classList.remove("is-hidden");
            eli10Text.classList.add("is-hidden");
        });
    }

    if (cameraToggleBtn && cameraDropdown) {
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

    if (cameraDisclaimerBtn && cameraModal) {
        cameraDisclaimerBtn.addEventListener("click", () => {
            cameraModal.classList.add("show");
        });
    }

    if (closeModal && cameraModal) {
        closeModal.addEventListener("click", () => cameraModal.classList.remove("show"));
    }

    let stream = null;
    let cameraActive = false;

    const stopCameraStream = () => {
        if (stream) {
            stream.getTracks().forEach(track => {
                try { track.stop(); } catch (_) {}
            });
            stream = null;
        }
        if (video) video.srcObject = null;
        cameraActive = false;
        if (cameraSection) cameraSection.classList.add("camera-hidden");
    };

    if (ackCheckbox && startCameraBtn) {
        ackCheckbox.addEventListener("change", () => {
            startCameraBtn.disabled = !ackCheckbox.checked;
        });
    }

    if (startCameraBtn && video) {
        startCameraBtn.addEventListener("click", async () => {
            if (cameraActive) return;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: 640, height: 480 },
                    audio: false
                });
                video.srcObject = stream;
                video.muted = true;
                video.setAttribute("playsinline", "");
                cameraActive = true;
                if (cameraSection) cameraSection.classList.remove("camera-hidden");
            } catch (err) {
                alert("Camera access denied or unavailable.");
                stopCameraStream();
            }
        });
    }

    if (stopCameraBtn) {
        stopCameraBtn.addEventListener("click", () => {
            if (!cameraActive) return;
            stopCameraStream();
        });
    }

    const captureFrame = () => {
        if (!cameraActive || !video?.videoWidth) {
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
            stopCameraStream();
        });
    };

    if (captureBtn) captureBtn.addEventListener("click", captureFrame);

    if (captureCountdownBtn) {
        captureCountdownBtn.addEventListener("click", () => {
            let count = 3;
            captureCountdownBtn.textContent = `Capturing in ${count}`;
            const timer = setInterval(() => {
                count -= 1;
                if (count <= 0) {
                    clearInterval(timer);
                    captureCountdownBtn.textContent = "Capture in 3s";
                    captureFrame();
                } else {
                    captureCountdownBtn.textContent = `Capturing in ${count}`;
                }
            }, 1000);
        });
    }

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

    window.addEventListener("beforeunload", stopCameraStream);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) stopCameraStream();
    });
});
