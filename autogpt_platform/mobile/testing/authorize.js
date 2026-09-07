document.querySelector("#connect").addEventListener("click", async () => {
  const button = document.querySelector("#connect");
  const result = document.querySelector("#authorization-result");
  const query = new URLSearchParams(location.search);
  button.disabled = true;
  result.textContent = "Returning to AutoGPT…";
  try {
    const response = await fetch("/api/auth/mobile/authorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code_challenge: query.get("code_challenge"),
        state: query.get("state"),
      }),
    });
    if (!response.ok) throw new Error("Fixture authorization failed");
    const authorization = await response.json();
    location.href = authorization.url;
  } catch {
    result.textContent = "Could not complete fixture authorization. Try again.";
    button.disabled = false;
  }
});
