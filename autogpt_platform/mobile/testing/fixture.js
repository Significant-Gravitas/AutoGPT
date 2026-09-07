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
