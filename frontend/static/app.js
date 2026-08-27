const decisionEl = document.getElementById("decision");
const detectionsEl = document.getElementById("detections");
const servoStateEl = document.getElementById("servoState");
const modePill = document.getElementById("modePill");
const healthLine = document.getElementById("healthLine");
const autoSort = document.getElementById("autoSort");
const conf = document.getElementById("conf");
const confVal = document.getElementById("confVal");

function renderStatus(data) {
  const decision = data.decision || "waiting…";
  decisionEl.textContent = decision;
  decisionEl.className = `decision ${data.decision || "idle"}`;

  detectionsEl.innerHTML = "";
  (data.detections || []).forEach((d) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${d.label} <small>(${d.source})</small></span><strong>${(d.confidence * 100).toFixed(0)}%</strong>`;
    detectionsEl.appendChild(li);
  });
  if (!(data.detections || []).length) {
    detectionsEl.innerHTML = `<li><span class="muted">No objects</span></li>`;
  }

  const servo = data.servo || {};
  servoStateEl.textContent = `servo: ${servo.connected ? "connected" : "dry-run"} · ${servo.last_action || "—"}`;
  modePill.textContent = `mode: ${data.detector_mode || "—"}`;
  autoSort.checked = !!data.auto_sort;
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthLine.textContent = `camera ${data.camera_mock ? "mock" : "live"} · frames via pipeline · models ${data.models_dir}`;
    modePill.textContent = `mode: ${data.detector_mode}`;
  } catch (err) {
    healthLine.textContent = "backend unreachable";
  }
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
  ws.onmessage = (ev) => {
    try {
      renderStatus(JSON.parse(ev.data));
    } catch (_) {}
  };
  ws.onclose = () => setTimeout(connectWs, 1500);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

document.querySelectorAll("[data-sort]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    await postJSON("/api/sort", { label: btn.dataset.sort });
  });
});

document.getElementById("homeBtn").addEventListener("click", async () => {
  await postJSON("/api/servo/home", {});
});

document.getElementById("reloadBtn").addEventListener("click", async () => {
  const data = await postJSON("/api/detector/reload", {});
  modePill.textContent = `mode: ${data.mode}`;
});

autoSort.addEventListener("change", async () => {
  await postJSON("/api/settings", { auto_sort: autoSort.checked });
});

conf.addEventListener("input", () => {
  confVal.textContent = conf.value;
});

conf.addEventListener("change", async () => {
  await postJSON("/api/settings", { conf_threshold: Number(conf.value) });
});

refreshHealth();
connectWs();
setInterval(refreshHealth, 8000);
