let stream = null;

const ackCheckbox = document.getElementById("ackCheckbox");
const startCameraBtn = document.getElementById("startCameraBtn");
const cameraSection = document.getElementById("cameraSection");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const previewSection = document.getElementById("previewSection");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const uploadForm = document.getElementById("uploadForm");

ackCheckbox.addEventListener("change", () => {
    startCameraBtn.disabled = !ackCheckbox.checked;
});

startCameraBtn.addEventListener("click", () => {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            cameraSection.style.display = "block";
            startCameraBtn.style.display = "none";
        });
});

document.getElementById("captureBtn").addEventListener("click", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const file = new File([blob], "camera.png", { type: "image/png" });
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        previewImage.src = URL.createObjectURL(blob);
        previewSection.style.display = "block";
    });

    stream.getTracks().forEach(t => t.stop());
});

document.getElementById("submitPreviewBtn").addEventListener("click", () => {
    uploadForm.submit();
});

/* -------- EyeGPT (RULE-BASED, INSTANT) -------- */

function askEyeGPT() {
    const box = document.getElementById("aiResponse");

    if (!window.lastResult) {
        box.innerHTML = "Please analyze an image first.";
        return;
    }

    box.innerHTML = "Generating explanation...";

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            result: window.lastResult
        })
    })
    .then(res => res.json())
    .then(data => {
        box.innerHTML = data.response.replace(/\n/g, "<br>");
    })
    .catch(() => {
        box.innerHTML = "EyeGPT is unavailable.";
    });
}

/* -------- Grad-CAM Toggle (FIXED) -------- */

function toggleGradCam(button) {
    const img = document.getElementById("gradcamImage");

    if (!img) return;

    if (img.style.display === "none") {
        img.style.display = "block";
        button.innerText = "Hide";
    } else {
        img.style.display = "none";
        button.innerText = "Show";
    }
}
