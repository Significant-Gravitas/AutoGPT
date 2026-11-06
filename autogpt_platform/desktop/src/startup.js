"use strict";

const message = document.getElementById("message");

window.autogpt.onRuntimeEvent((event) => {
  if (event.event === "progress" && event.message) {
    message.textContent = event.message;
  } else if (event.event === "ready") {
    message.textContent = "Opening AutoGPT…";
  } else if (event.event === "error") {
    message.textContent = event.message || "Something went wrong.";
    if (event.fatal) document.body.classList.add("failed");
  }
});

// Shown with a failed start only (#actions): a newer version may be the fix.
const update = document.getElementById("update");
window.autogpt.onUpdateOffer((offer) => {
  update.hidden = !offer;
  if (!offer) return;
  update.textContent = offer.label;
  update.disabled = !offer.action;
});
update.addEventListener("click", () => window.autogpt.takeUpdateOffer());

document.getElementById("logs").addEventListener("click", () => window.autogpt.openLogs());
document.getElementById("quit").addEventListener("click", () => window.autogpt.quit());
document.getElementById("doctor").addEventListener("click", () => window.autogpt.openDoctor());
