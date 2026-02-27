const form = document.getElementById("predict-form");
const responseBox = document.getElementById("response");
const jsonButton = document.getElementById("send-json");

async function showResponse(response) {
  const data = await response.json();
  responseBox.textContent = JSON.stringify(data, null, 2);
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    await showResponse(response);
  });
}

if (jsonButton) {
  jsonButton.addEventListener("click", async () => {
    const payload = {
      event: "fall_candidate",
      confidence: 0.42,
    };

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    await showResponse(response);
  });
}

