async function refreshSession() {
  const result = document.querySelector("#session-result");
  try {
    const response = await fetch("/api/fixture/session", {
      credentials: "same-origin",
    });
    const session = await response.json();
    result.textContent = session.authenticated
      ? session.cacheCookieReceived
        ? "Fixture session connected · token + cache cookies received"
        : "Fixture session connected · cache cookie missing"
      : "No fixture session · use native Sign in";
    result.dataset.connected = String(session.authenticated);
  } catch {
    result.textContent = "Fixture server unreachable";
  }
  document.querySelector("#cookie-result").textContent =
    document.cookie.includes("better-auth.session_")
      ? "Unexpected: fixture session is visible to page JavaScript."
      : "HttpOnly check: fixture session cookie is not readable by page JavaScript.";
}

function showFiles(event) {
  const names = Array.from(
    event.target.files,
    (file) => `${file.name} (${file.size} bytes)`,
  );
  document.querySelector("#file-result").textContent = names.length
    ? names.join(" · ")
    : "Selection canceled or empty.";
}

document
  .querySelector("#refresh-session")
  .addEventListener("click", refreshSession);
document.querySelector("#clear-session").addEventListener("click", async () => {
  await fetch("/api/fixture/logout", { method: "POST" });
  await refreshSession();
});
document.querySelector("#single-file").addEventListener("change", showFiles);
document.querySelector("#multiple-files").addEventListener("change", showFiles);
document
  .querySelector("#popup")
  .addEventListener("click", () =>
    window.open("https://example.com/", "_blank", "noopener"),
  );
document.querySelector("#location-result").textContent =
  `Current route: ${location.pathname}${location.search}`;
document.querySelector("#transport-result").textContent =
  `Attachment transport: ${location.protocol === "https:" ? "HTTPS" : "local HTTP"}. Repeat with the optional trusted HTTPS fixture to compare download policy.`;
refreshSession();

document.querySelector("#blob-download").addEventListener("click", () => {
  const blob = new Blob(
    [
      "# Native integration fixture\n\nGenerated download; no live chat data.\n",
    ],
    { type: "text/markdown;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "native-fixture-generated.md";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
});

let streamController;
let lastStreamDropped = false;

async function runStream(drop = false) {
  streamController?.abort();
  const controller = new AbortController();
  streamController = controller;
  lastStreamDropped = drop;
  const result = document.querySelector("#stream-result");
  const text = document.querySelector("#stream-text");
  const startedAt = performance.now();
  let received = 0;
  let textDeliveries = 0;
  let completed = false;
  let reader;
  result.textContent = drop
    ? "Opening fixture stream · intentional drop after chunk 2…"
    : "Opening fixture stream…";
  result.dataset.connected = "false";
  text.textContent = "";
  for (const id of ["stream-start", "stream-drop", "stream-retry"])
    document.getElementById(id).disabled = true;
  document.querySelector("#stream-cancel").disabled = false;
  try {
    const response = await fetch(
      `/api/fixture/stream${drop ? "?drop=1" : ""}`,
      {
        signal: controller.signal,
        cache: "no-store",
      },
    );
    if (!response.ok || !response.body)
      throw new Error("Streaming response is unavailable.");
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const chunk = await reader.read();
      if (streamController !== controller) return;
      if (chunk.done) break;
      const previousCount = received;
      buffer += decoder.decode(chunk.value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = frame.split("\n");
        const event = lines
          .find((line) => line.startsWith("event: "))
          ?.slice(7);
        const data = JSON.parse(
          lines.find((line) => line.startsWith("data: "))?.slice(6) ?? "null",
        );
        if (data?.fixture !== true) throw new Error("Unexpected stream data.");
        if (event === "chunk" && data.index === received + 1) {
          received++;
          const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
          text.textContent += `${received} · ${elapsed}s · ${data.text}\n`;
          result.textContent = `Receiving fixture text · ${received}/5 chunks · response still open`;
        } else if (event === "done" && data.chunks === received) {
          completed = true;
        } else throw new Error("Unexpected stream sequence.");
      }
      if (received > previousCount) textDeliveries++;
    }
    if (!completed) throw new Error("Stream ended without a completion event.");
    result.textContent = `Complete · ${received}/5 fixture chunks · ${textDeliveries} text deliveries · ${textDeliveries > 1 ? "incremental arrival" : "text buffered in one delivery"}`;
    result.dataset.connected = "true";
  } catch (error) {
    if (streamController !== controller) return;
    result.textContent = controller.signal.aborted
      ? `Canceled · ${received}/5 chunks received. Retry starts a fresh request.`
      : `Stream interrupted · ${received}/5 chunks received; partial text retained. ${drop ? "This disconnect is intentional." : error.message} Retry is available.`;
  } finally {
    reader?.releaseLock();
    if (streamController === controller) {
      streamController = undefined;
      for (const id of ["stream-start", "stream-drop", "stream-retry"])
        document.getElementById(id).disabled = false;
      document.querySelector("#stream-cancel").disabled = true;
    }
  }
}

document
  .querySelector("#stream-start")
  .addEventListener("click", () => runStream());
document
  .querySelector("#stream-drop")
  .addEventListener("click", () => runStream(true));
document
  .querySelector("#stream-retry")
  .addEventListener("click", () => runStream(lastStreamDropped));
document
  .querySelector("#stream-cancel")
  .addEventListener("click", () => streamController?.abort());
window.addEventListener("pagehide", () => streamController?.abort());

document
  .querySelector("#microphone")
  .addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const result = document.querySelector("#microphone-result");
    if (!navigator.mediaDevices?.getUserMedia) {
      result.textContent =
        "Microphone API unavailable. Use a secure context and a supported webview.";
      return;
    }
    button.disabled = true;
    result.textContent = "Waiting for microphone permission…";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const track of stream.getTracks()) track.stop();
      result.textContent =
        "Allowed · all tracks stopped immediately. No audio recorded or uploaded.";
    } catch (error) {
      result.textContent =
        error.name === "NotAllowedError"
          ? "Denied · microphone permission was not granted. No audio recorded or uploaded."
          : `Microphone unavailable (${error.name}). No audio recorded or uploaded.`;
    } finally {
      button.disabled = false;
    }
  });
