const classificationSelect = document.getElementById("classification-model");
const keypointSelect = document.getElementById("keypoint-model");
const videoSelect = document.getElementById("video-select");
const runButton = document.getElementById("run-inference");
const responseBox = document.getElementById("response");
const statusLabel = document.getElementById("run-status");

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

function renderResult(data) {
  if (!responseBox) {
    return;
  }

  const command = data.command || "";
  const stdout = data.stdout || "(empty)";
  const stderr = data.stderr || "(empty)";
  const returncode = Number.isInteger(data.returncode) ? data.returncode : "N/A";
  const ok = Boolean(data.ok) && returncode === 0;

  responseBox.textContent = [
    `Return Code: ${returncode}`,
    "",
    "Command:",
    command,
    "",
    "STDOUT:",
    stdout,
    "",
    "STDERR:",
    stderr,
  ].join("\n");

  setOutputState(ok);
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
      if (responseBox) {
        responseBox.textContent = "No videos found in Datasets\\test_vids.";
        responseBox.classList.add("output-error");
      }
      return;
    }

    videoSelect.innerHTML = "";
    for (const videoName of videos) {
      const option = document.createElement("option");
      option.value = videoName;
      option.textContent = videoName;
      videoSelect.appendChild(option);
    }

    runButton.disabled = false;
  } catch (error) {
    if (statusLabel) {
      statusLabel.textContent = "Error";
      statusLabel.classList.add("status-error");
    }
    if (responseBox) {
      responseBox.textContent = `Failed to load videos: ${error.message}`;
      responseBox.classList.add("output-error");
    }
  }
}

async function runInference() {
  if (!classificationSelect || !keypointSelect || !videoSelect || !runButton || !responseBox || !statusLabel) {
    return;
  }

  const payload = {
    classification_model: classificationSelect.value,
    keypoint_model: keypointSelect.value,
    video: videoSelect.value,
  };

  const previousLabel = runButton.textContent;
  runButton.disabled = true;
  runButton.textContent = "Running...";
  statusLabel.textContent = "Running...";
  statusLabel.classList.remove("status-success", "status-error");
  responseBox.classList.remove("output-success", "output-error");
  responseBox.textContent = "Running inference...";

  try {
    const response = await fetch("/api/run_inference", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok && !data.ok) {
      renderResult(data);
      return;
    }
    renderResult(data);
  } catch (error) {
    renderResult({
      ok: false,
      command: "",
      returncode: "N/A",
      stdout: "",
      stderr: error.message || "Request failed.",
    });
  } finally {
    runButton.disabled = !videoSelect.value;
    runButton.textContent = previousLabel;
  }
}

if (runButton) {
  runButton.addEventListener("click", runInference);
}

loadTestVideos();
