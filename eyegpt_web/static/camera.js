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
        })
        .catch(() => alert("Camera access denied or unavailable."));
});

document.getElementById("captureBtn").addEventListener("click", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const file = new File([blob], "camera_demo.png", { type: "image/png" });
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
