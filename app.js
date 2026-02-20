import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// DOM
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const video = document.getElementById("webcam");
const desktop = document.getElementById("desktop");
const overlay = document.getElementById("overlay");
const logTextarea = document.getElementById("logs");

const files = Array.from(document.querySelectorAll(".file"));

const overlayCtx = overlay.getContext("2d");
const drawingUtils = new DrawingUtils(overlayCtx);

// State
let handLandmarker = null;
let running = false;

let lastVideoTime = -1;
let isPinching = false;
let wasPinching = false;
let activeFile = null;

// =========================
// 1) One Euro filter
// =========================
class OneEuroFilter {
  constructor(freq = 60, minCutoff = 1.2, beta = 0.05, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;

    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  lowpass(prev, curr, a) {
    return prev + a * (curr - prev);
  }

  filter(x, tMs) {
    if (this.tPrev == null) {
      this.tPrev = tMs;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }

    const dt = Math.max(1e-6, (tMs - this.tPrev) / 1000);
    this.freq = 1.0 / dt;

    const dx = (x - this.xPrev) / dt;
    const ad = this.alpha(this.dCutoff);
    const dxHat = this.lowpass(this.dxPrev, dx, ad);

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff);

    const xHat = this.lowpass(this.xPrev, x, a);

    this.tPrev = tMs;
    this.xPrev = xHat;
    this.dxPrev = dxHat;

    return xHat;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

const xFilter = new OneEuroFilter(60, 1.2, 0.05, 1.0);
const yFilter = new OneEuroFilter(60, 1.2, 0.05, 1.0);

// =========================
// 2) Pinch debounce + hysteresis
// =========================
let pinchClosed = false;
let pinchOnCount = 0;
let pinchOffCount = 0;

const PINCH_ON = 0.30;
const PINCH_OFF = 0.60;
const ON_FRAMES = 2;
const OFF_FRAMES = 3;

// Utils
function log(msg) {
  const ts = new Date().toISOString().split("T")[1].split(".")[0];
  logTextarea.value = `[${ts}] ${msg}\n` + logTextarea.value;
}

function setStatus(msg) {
  statusEl.textContent = msg;
  log(msg);
}

function resizeOverlay() {
  const rect = desktop.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// rough hand size: wrist ↔ index MCP
function getHandScale(landmarks) {
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const d = distance(wrist, indexMcp);
  return d > 0 ? d : 0.0001;
}

// Mirror X so hand movement feels natural left/right
function screenCoordsFromLandmark(landmark) {
  const rect = desktop.getBoundingClientRect();
  const mirroredX = 1 - landmark.x;
  const x = mirroredX * rect.width + rect.left;
  const y = landmark.y * rect.height + rect.top;
  return { x, y };
}

function getFileUnderPoint(x, y) {
  for (let i = files.length - 1; i >= 0; i--) {
    const file = files[i];
    const rect = file.getBoundingClientRect();
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      return file;
    }
  }
  return null;
}

function updateFilePosition(file, x, y) {
  const desktopRect = desktop.getBoundingClientRect();
  const fileRect = file.getBoundingClientRect();
  const offsetX = fileRect.width / 2;
  const offsetY = fileRect.height / 2;

  let newLeft = x - desktopRect.left - offsetX;
  let newTop = y - desktopRect.top - offsetY;

  newLeft = Math.max(0, Math.min(newLeft, desktopRect.width - fileRect.width));
  newTop = Math.max(0, Math.min(newTop, desktopRect.height - fileRect.height));

  file.style.left = newLeft + "px";
  file.style.top = newTop + "px";
}

function updatePinchStateDebounced(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];

  const rawDist = distance(thumbTip, indexTip);
  const handScale = getHandScale(landmarks);
  const normDist = rawDist / handScale;

  if (!pinchClosed) {
    if (normDist < PINCH_ON) pinchOnCount++;
    else pinchOnCount = 0;

    if (pinchOnCount >= ON_FRAMES) {
      pinchClosed = true;
      pinchOnCount = 0;
      pinchOffCount = 0;
      log(`Pinch ON (normDist=${normDist.toFixed(2)})`);
    }
  } else {
    if (normDist > PINCH_OFF) pinchOffCount++;
    else pinchOffCount = 0;

    if (pinchOffCount >= OFF_FRAMES) {
      pinchClosed = false;
      pinchOffCount = 0;
      pinchOnCount = 0;
      log(`Pinch OFF (normDist=${normDist.toFixed(2)})`);
    }
  }

  return pinchClosed;
}

function drawLandmarks(results) {
  overlayCtx.save();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results?.landmarks?.length) {
    overlayCtx.restore();
    return;
  }

  const h = results.landmarks[0];
  const mirrored = h.map((p) => ({ ...p, x: 1 - p.x }));

  drawingUtils.drawConnectors(mirrored, HandLandmarker.HAND_CONNECTIONS);
  drawingUtils.drawLandmarks(mirrored, { radius: 3 });

  overlayCtx.restore();
}

// MediaPipe init/start
async function initHandLandmarker() {
  setStatus("Loading MediaPipe models…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    },
    numHands: 1,
    runningMode: "VIDEO"
  });
  setStatus("Models loaded. Click 'Start Camera'.");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });

  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);

  running = true;
  setStatus("Camera started. Show your hand in front of the camera.");
  log("Ready – pinch over a file to drag it.");

  window.requestAnimationFrame(loop);
}

// Main loop
function loop() {
  if (!running || !handLandmarker) return;

  const videoTime = video.currentTime;
  if (videoTime === lastVideoTime) {
    window.requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = videoTime;

  const nowMs = performance.now();
  const results = handLandmarker.detectForVideo(video, nowMs);

  drawLandmarks(results);

  if (results?.landmarks?.length) {
    const landmarks = results.landmarks[0];

    const pinchNow = updatePinchStateDebounced(landmarks);

    const cursorNorm = landmarks[8]; // index fingertip
    let { x, y } = screenCoordsFromLandmark(cursorNorm);

    // OneEuro smoothing
    x = xFilter.filter(x, nowMs);
    y = yFilter.filter(y, nowMs);

    isPinching = pinchNow;

    if (isPinching && !wasPinching) {
      const file = getFileUnderPoint(x, y);
      if (file) {
        activeFile = file;
        activeFile.classList.add("active");
        log("Pinch start on " + activeFile.dataset.id);
      } else {
        activeFile = null;
      }
    }

    if (isPinching && activeFile) {
      updateFilePosition(activeFile, x, y);
    }

    if (!isPinching && wasPinching && activeFile) {
      log("Pinch released – dropped " + activeFile.dataset.id);
      activeFile.classList.remove("active");
      activeFile = null;
    }

    wasPinching = isPinching;
  } else {
    // hand lost
    isPinching = false;

    if (wasPinching && activeFile) {
      log("Hand lost – dropping " + activeFile.dataset.id);
      activeFile.classList.remove("active");
      activeFile = null;
    }

    wasPinching = false;

    pinchClosed = false;
    pinchOnCount = 0;
    pinchOffCount = 0;

    xFilter.reset();
    yFilter.reset();
  }

  window.requestAnimationFrame(loop);
}

// Wire up
startBtn.addEventListener("click", () => {
  if (!handLandmarker) {
    setStatus("Still loading models – please wait a moment…");
    return;
  }
  if (!running) startCamera().catch((err) => setStatus("Error starting camera: " + err.message));
});

if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
  setStatus("getUserMedia not supported in this browser.");
} else {
  initHandLandmarker().catch((err) => setStatus("Failed to init MediaPipe: " + err.message));
}
