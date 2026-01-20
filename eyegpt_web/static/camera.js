const cameraToggleBtn = document.getElementById("cameraToggleBtn");
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

let stream = null;

/* ===============================
   TOGGLE CAMERA (SAFE)
================================ */
if (cameraToggleBtn && cameraSection) {
    cameraToggleBtn.onclick = () => {
        cameraSection.classList.toggle("camera-hidden");
    };
}

/* ===============================
   ACK GATE
================================ */
if (ackCheckbox && startCameraBtn) {
    ackCheckbox.onchange = () => {
        startCameraBtn.disabled = !ackCheckbox.checked;
    };
}

/* ===============================
   START CAMERA
================================ */
if (startCameraBtn && video) {
    startCameraBtn.onclick = async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user" }
            });

            video.srcObject = stream;

            /* 🔹 ADDITION: prevent video hijacking layout */
            video.style.maxWidth = "100%";
            video.style.pointerEvents = "auto";

        } catch (err) {
            alert("Camera access denied or unavailable.");
        }
    };
}

/* ===============================
   CAPTURE FRAME
================================ */
if (captureBtn && canvas && video) {
    captureBtn.onclick = () => {
        if (!video.videoWidth || !video.videoHeight) {
            alert("Camera not ready.");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        canvas.toBlob(blob => {
            if (!blob) return;

            const file = new File([blob], "camera.png", { type: "image/png" });
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;

            previewImage.src = URL.createObjectURL(blob);
            previewSection.style.display = "block";
        });

        /* 🔹 ADDITION: cleanly stop stream */
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
    };
}

/* ===============================
   SUBMIT PREVIEW
================================ */
if (submitPreviewBtn && uploadForm) {
    submitPreviewBtn.onclick = () => uploadForm.submit();
}
