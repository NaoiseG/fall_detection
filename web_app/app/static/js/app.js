const classificationSelect = document.getElementById("classification-model");
const keypointSelect = document.getElementById("keypoint-model");
const keypointPrecisionSelect = document.getElementById("keypoint-precision");
const videoSelect = document.getElementById("video-select");
const cameraSelectLabel = document.getElementById("camera-select-label");
const cameraSelect = document.getElementById("camera-select");
const windowSizeInput = document.getElementById("window-size");
const strideOverlapInput = document.getElementById("stride-overlap");
const samplingKInput = document.getElementById("sampling-k");
const displayFpsInput = document.getElementById("display-fps");
const saveOutputInput = document.getElementById("save-output");
const runButton = document.getElementById("run-inference");
const stopButton = document.getElementById("stop-inference");
const responseBox = document.getElementById("response");
const statusLabel = document.getElementById("run-status");
const liveStatus = document.getElementById("live-status");
const liveCanvas = document.getElementById("live-canvas");

const frameQueue = [];
let queueProcessing = false;
let activeEventSource = null;
let streamTerminated = false;
let activeStatusUrl = "";
let activeJobId = "";
let activeStopUrl = "";
let isLiveMode = false;

function setOutputState(ok) {
  if (!responseBox || !statusLabel) {
    return;
  }

  responseBox.classList.remove("output-success", "output-error");
  statusLabel.classList.remove("status-success", "status-error");

  if (ok) {
    responseBox.classList.add("output-success");
    statusLabel.classList.add("status-success");
    statusLabel.textContent = "Success";
  } else {
    responseBox.classList.add("output-error");
    statusLabel.classList.add("status-error");
    statusLabel.textContent = "Error";
  }
}

function setLiveStatus(text) {
  if (liveStatus) {
    liveStatus.textContent = text;
  }
}

function isLiveSelected() {
  return videoSelect && videoSelect.value === "live";
}

function setRunReady() {
  if (!runButton || !videoSelect) {
    return;
  }
  if (isLiveSelected()) {
    runButton.disabled = !cameraSelect || !cameraSelect.value;
  } else {
    runButton.disabled = !videoSelect.value;
  }
  runButton.textContent = "Run";
  if (stopButton) {
    stopButton.disabled = false;
    stopButton.textContent = "Stop";
    stopButton.style.display = "none";
  }
}

function syncPrecisionControl() {
  if (!keypointSelect || !keypointPrecisionSelect) {
    return;
  }

  const fp16Supported = keypointSelect.value === "ultralytics-yolo11l";
  if (!fp16Supported) {
    keypointPrecisionSelect.value = "FP32";
  }
  keypointPrecisionSelect.disabled = !fp16Supported;
}

function clearFrameQueue() {
  frameQueue.length = 0;
}

function closeActiveStream() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  streamTerminated = false;
  activeStatusUrl = "";
  activeJobId = "";
  activeStopUrl = "";
  isLiveMode = false;
}

function renderStartInfo(data) {
  if (!responseBox) {
    return;
  }
  const lines = [
    "Inference stream started.",
    `job_id: ${data.job_id || "N/A"}`,
    `stream_url: ${data.stream_url || "N/A"}`,
    `status_url: ${data.status_url || "N/A"}`,
  ];
  if (data.stop_url) {
    lines.push(`stop_url: ${data.stop_url}`);
  }
  if (data.save_path) {
    lines.push(`saving_to: ${data.save_path}`);
  }
  responseBox.textContent = lines.join("\n");
}

function renderDoneInfo(data = {}) {
  if (!responseBox) {
    return;
  }
  const lines = ["Inference stream finished."];
  if (data.save_path) {
    lines.push(`saved_to: ${data.save_path}`);
  }
  responseBox.textContent = lines.join("\n");
}

function renderError(message) {
  if (responseBox) {
    responseBox.textContent = String(message || "Unknown error.");
  }
  setOutputState(false);
}

function parseIntInput(inputElement, name, minValue, maxValue = null) {
  const rawValue = String(inputElement?.value ?? "").trim();
  if (!rawValue) {
    throw new Error(`Missing ${name}.`);
  }
  if (!/^-?\d+$/.test(rawValue)) {
    throw new Error(`Invalid ${name}.`);
  }
  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value < minValue) {
    throw new Error(`Invalid ${name}.`);
  }
  if (maxValue !== null && value > maxValue) {
    throw new Error(`Invalid ${name}.`);
  }
  return value;
}

function readInferenceKnobs() {
  if (!windowSizeInput || !strideOverlapInput || !samplingKInput || !displayFpsInput) {
    throw new Error("Inference knobs are not available.");
  }

  const windowSize = parseIntInput(windowSizeInput, "window size (T)", 1);
  const overlapPercent = parseIntInput(strideOverlapInput, "overlap (%)", 0, 99);
  const samplingK = parseIntInput(samplingKInput, "sampling frequency (k)", 1);
  const displayFps = parseIntInput(displayFpsInput, "FPS", 1);
  return {
    T: windowSize,
    overlapPercent,
    k: samplingK,
    fps: displayFps,
  };
}

async function decodeFrameImage(frameJpegB64) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode frame JPEG."));
    image.src = `data:image/jpeg;base64,${frameJpegB64}`;
  });
}

function drawHudOverlay(ctx, hudLines) {
  if (!Array.isArray(hudLines) || hudLines.length === 0) {
    return;
  }

  const x = 10;
  const y = 10;
  const pad = 8;
  const lineGap = 6;

  ctx.save();
  ctx.font = "20px sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";

  const lineMetrics = hudLines.map((line) => {
    const m = ctx.measureText(String(line));
    const ascent = Number(m.actualBoundingBoxAscent) || 16;
    const descent = Number(m.actualBoundingBoxDescent) || 4;
    return {
      text: String(line),
      width: m.width,
      height: ascent + descent,
      ascent,
    };
  });

  const maxWidth = Math.max(...lineMetrics.map((m) => m.width), 0);
  const totalHeight =
    lineMetrics.reduce((sum, m) => sum + m.height, 0) +
    Math.max(0, lineMetrics.length - 1) * lineGap;

  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(x, y, maxWidth + pad * 2, totalHeight + pad * 2);

  ctx.fillStyle = "rgb(255, 255, 255)";
  let yCursor = y + pad;
  for (const line of lineMetrics) {
    yCursor += line.ascent;
    ctx.fillText(line.text, x + pad, yCursor);
    yCursor += line.height - line.ascent + lineGap;
  }
  ctx.restore();
}

function drawPoseOverlay(ctx, pose) {
  if (!pose || !Array.isArray(pose.xy) || !Array.isArray(pose.conf)) {
    return;
  }

  const xy = pose.xy;
  const conf = pose.conf;
  const threshold = Number(pose.conf_thres);
  const confThreshold = Number.isFinite(threshold) ? threshold : 0.2;
  const skeleton = Array.isArray(pose.skeleton) ? pose.skeleton : [];

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgb(255, 255, 0)";
  for (const edge of skeleton) {
    if (!Array.isArray(edge) || edge.length < 2) {
      continue;
    }
    const a = Number(edge[0]);
    const b = Number(edge[1]);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      continue;
    }
    if (a < 0 || b < 0 || a >= conf.length || b >= conf.length) {
      continue;
    }
    const confA = Number(conf[a]);
    const confB = Number(conf[b]);
    if (!(confA > confThreshold && confB > confThreshold)) {
      continue;
    }
    const pointA = xy[a];
    const pointB = xy[b];
    if (!Array.isArray(pointA) || !Array.isArray(pointB) || pointA.length < 2 || pointB.length < 2) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(Number(pointA[0]), Number(pointA[1]));
    ctx.lineTo(Number(pointB[0]), Number(pointB[1]));
    ctx.stroke();
  }

  ctx.fillStyle = "rgb(0, 255, 0)";
  for (let i = 0; i < xy.length; i += 1) {
    const point = xy[i];
    if (!Array.isArray(point) || point.length < 2) {
      continue;
    }
    const score = Number(conf[i]);
    if (!(score > confThreshold)) {
      continue;
    }
    ctx.beginPath();
    ctx.arc(Number(point[0]), Number(point[1]), 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

async function drawPacket(packet) {
  if (!liveCanvas || !packet || typeof packet.frame_jpeg_b64 !== "string") {
    return;
  }
  const ctx = liveCanvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const image = await decodeFrameImage(packet.frame_jpeg_b64);
  const targetW = Number(packet?.size?.w) || image.width;
  const targetH = Number(packet?.size?.h) || image.height;
  if (targetW > 0 && targetH > 0 && (liveCanvas.width !== targetW || liveCanvas.height !== targetH)) {
    liveCanvas.width = targetW;
    liveCanvas.height = targetH;
  }

  ctx.drawImage(image, 0, 0, liveCanvas.width, liveCanvas.height);
  drawPoseOverlay(ctx, packet.pose);
  drawHudOverlay(ctx, packet.hud_lines);

  const frameNumber = Number(packet.frame_number) || 0;
  const frameCount = Number(packet.frame_count) || 0;
  const fps = Number(packet.fps);
  const fpsText = Number.isFinite(fps) && fps > 0 ? fps.toFixed(1) : "NA";
  if (frameCount > 0) {
    setLiveStatus(`Running... frame ${frameNumber}/${frameCount} | fps ${fpsText}`);
  } else {
    setLiveStatus(`Running... frame ${frameNumber} | fps ${fpsText}`);
  }
}

async function processFrameQueue() {
  if (queueProcessing) {
    return;
  }
  queueProcessing = true;
  try {
    while (frameQueue.length > 0) {
      const packet = frameQueue.shift();
      await drawPacket(packet);
    }
  } catch (error) {
    renderError(error.message || "Failed while rendering streamed frames.");
    setLiveStatus("Error.");
    closeActiveStream();
    setRunReady();
  } finally {
    queueProcessing = false;
  }
}

async function fetchJobStatus() {
  if (!activeStatusUrl) {
    return null;
  }
  try {
    const response = await fetch(activeStatusUrl);
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (_error) {
    return null;
  }
}

function attachStreamHandlers(streamUrl, statusUrl) {
  closeActiveStream();
  clearFrameQueue();
  streamTerminated = false;
  activeStatusUrl = statusUrl || "";

  activeEventSource = new EventSource(streamUrl);

  activeEventSource.addEventListener("frame", (event) => {
    try {
      const packet = JSON.parse(event.data);
      frameQueue.push(packet);
      void processFrameQueue();
    } catch (error) {
      renderError(error.message || "Invalid frame packet.");
      setLiveStatus("Error.");
      closeActiveStream();
      setRunReady();
    }
  });

  activeEventSource.addEventListener("heartbeat", () => {
    if (liveStatus && liveStatus.textContent === "Starting inference...") {
      setLiveStatus("Running...");
    }
  });

  activeEventSource.addEventListener("done", (event) => {
    streamTerminated = true;
    setOutputState(true);
    setLiveStatus("Done.");
    try {
      renderDoneInfo(JSON.parse(event.data || "{}"));
    } catch (_parseError) {
      renderDoneInfo();
    }
    closeActiveStream();
    setRunReady();
  });

  activeEventSource.addEventListener("error", (event) => {
    const hasData = typeof event?.data === "string" && event.data.length > 0;
    if (!hasData) {
      return;
    }
    streamTerminated = true;
    try {
      const payload = JSON.parse(event.data);
      renderError(payload.message || "Inference failed.");
    } catch (_parseError) {
      renderError("Inference failed.");
    }
    setLiveStatus("Error.");
    closeActiveStream();
    setRunReady();
  });

  activeEventSource.onerror = async () => {
    if (streamTerminated) {
      return;
    }

    const status = await fetchJobStatus();
    if (status && status.status === "done") {
      streamTerminated = true;
      setOutputState(true);
      setLiveStatus("Done.");
      renderDoneInfo(status);
    } else {
      const message = status?.error || "Stream disconnected.";
      renderError(message);
      setLiveStatus("Error.");
    }
    closeActiveStream();
    setRunReady();
  };
}

async function stopLiveInference() {
  if (!activeStopUrl) {
    return;
  }
  try {
    await fetch(activeStopUrl, { method: "POST" });
  } catch (_err) {
    // best-effort stop
  }
}

async function loadCameras() {
  if (!cameraSelect) {
    return;
  }
  cameraSelect.innerHTML = '<option value="">Loading cameras...</option>';
  if (runButton) {
    runButton.disabled = true;
  }
  try {
    const response = await fetch("/api/list_cameras");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load cameras.");
    }
    const cameras = Array.isArray(data.cameras) ? data.cameras : [];
    if (cameras.length === 0) {
      cameraSelect.innerHTML = '<option value="">No cameras found</option>';
      renderError("No cameras found. Plug in a camera and refresh.");
      return;
    }
    cameraSelect.innerHTML = "";
    for (const cam of cameras) {
      const option = document.createElement("option");
      option.value = String(cam.index);
      option.textContent = `${cam.label} (${cam.width}x${cam.height})`;
      cameraSelect.appendChild(option);
    }
    setRunReady();
  } catch (error) {
    cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
    renderError(`Failed to load cameras: ${error.message}`);
  }
}

function showCameraSelector() {
  if (cameraSelectLabel) cameraSelectLabel.style.display = "";
  if (cameraSelect) cameraSelect.style.display = "";
}

function hideCameraSelector() {
  if (cameraSelectLabel) cameraSelectLabel.style.display = "none";
  if (cameraSelect) cameraSelect.style.display = "none";
}

async function loadTestVideos() {
  if (!videoSelect || !runButton) {
    return;
  }

  runButton.disabled = true;
  videoSelect.innerHTML = '<option value="">Loading videos...</option>';

  try {
    const response = await fetch("/api/list_test_videos");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load test videos.");
    }

    const videos = Array.isArray(data.videos) ? data.videos : [];
    if (videos.length === 0) {
      videoSelect.innerHTML = '<option value="">No videos found</option>';
      if (statusLabel) {
        statusLabel.textContent = "Error";
        statusLabel.classList.add("status-error");
      }
      renderError("No videos found in Datasets\\test_vids.");
      return;
    }

    videoSelect.innerHTML = "";
    for (const videoName of videos) {
      const option = document.createElement("option");
      option.value = videoName;
      option.textContent = videoName;
      videoSelect.appendChild(option);
    }
    // Live mode sentinel at the bottom of the list.
    const liveOption = document.createElement("option");
    liveOption.value = "live";
    liveOption.textContent = "--- Live (webcam) ---";
    videoSelect.appendChild(liveOption);

    setRunReady();
  } catch (error) {
    if (statusLabel) {
      statusLabel.textContent = "Error";
      statusLabel.classList.add("status-error");
    }
    renderError(`Failed to load videos: ${error.message}`);
  }
}

async function runInference() {
  if (
    !classificationSelect ||
    !keypointSelect ||
    !videoSelect ||
    !windowSizeInput ||
    !strideOverlapInput ||
    !samplingKInput ||
    !displayFpsInput ||
    !runButton ||
    !responseBox ||
    !statusLabel
  ) {
    return;
  }

  closeActiveStream();
  clearFrameQueue();

  let knobs;
  try {
    knobs = readInferenceKnobs();
  } catch (error) {
    renderError(error.message || "Invalid inference settings.");
    setLiveStatus("Error.");
    setRunReady();
    return;
  }

  const live = isLiveSelected();
  const payload = {
    classification_model: classificationSelect.value,
    keypoint_model: keypointSelect.value,
    keypoint_precision: keypointPrecisionSelect ? keypointPrecisionSelect.value : "FP32",
    video: videoSelect.value,
    save_output: saveOutputInput ? saveOutputInput.checked : false,
    T: knobs.T,
    stride: knobs.overlapPercent,
    overlap_percent: knobs.overlapPercent,
    k: knobs.k,
    display_fps: knobs.fps,
  };
  if (live && cameraSelect) {
    payload.camera_index = parseInt(cameraSelect.value, 10) || 0;
  }

  runButton.disabled = true;
  runButton.textContent = "Running...";
  statusLabel.textContent = "Running...";
  statusLabel.classList.remove("status-success", "status-error");
  responseBox.classList.remove("output-success", "output-error");
  responseBox.textContent = saveOutputInput && saveOutputInput.checked
    ? "Starting inference stream with save mode enabled..."
    : "Starting inference stream...";
  setLiveStatus("Starting inference...");

  try {
    const response = await fetch("/api/start_inference_stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      renderError(data.error || "Failed to start inference stream.");
      setLiveStatus("Error.");
      setRunReady();
      return;
    }

    renderStartInfo(data);
    activeJobId = data.job_id || "";
    activeStopUrl = data.stop_url || "";
    isLiveMode = live;
    if (live && stopButton) {
      stopButton.style.display = "";
    }
    attachStreamHandlers(data.stream_url, data.status_url);
  } catch (error) {
    renderError(error.message || "Request failed.");
    setLiveStatus("Error.");
    setRunReady();
  }
}

if (runButton) {
  runButton.addEventListener("click", () => {
    void runInference();
  });
}

if (videoSelect) {
  videoSelect.addEventListener("change", () => {
    if (activeEventSource) {
      return;
    }
    if (isLiveSelected()) {
      showCameraSelector();
      void loadCameras();
    } else {
      hideCameraSelector();
      setRunReady();
    }
  });
}

if (cameraSelect) {
  cameraSelect.addEventListener("change", () => {
    if (!activeEventSource) {
      setRunReady();
    }
  });
}

if (stopButton) {
  stopButton.addEventListener("click", () => {
    stopButton.disabled = true;
    stopButton.textContent = "Stopping...";
    void stopLiveInference();
    // The SSE stream will emit "done" after the job stops; the normal
    // done/error handlers will reset the UI.  If the stream has already
    // closed we fall back to a manual reset after a short delay.
    setTimeout(() => {
      if (!activeEventSource) {
        closeActiveStream();
        setRunReady();
      }
      if (stopButton) {
        stopButton.disabled = false;
      }
    }, 3000);
  });
}

if (keypointSelect) {
  keypointSelect.addEventListener("change", () => {
    syncPrecisionControl();
  });
}

syncPrecisionControl();
loadTestVideos();
