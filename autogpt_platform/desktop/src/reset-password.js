"use strict";

const form = document.getElementById("form");
const password = document.getElementById("password");
const confirmation = document.getElementById("confirmation");
const message = document.getElementById("message");
const submit = document.getElementById("submit");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  message.textContent = "";
  const { error } = await window.autogpt.resetOwnerPassword(password.value, confirmation.value);
  if (!error) return; // the window closes and AutoGPT restarts
  message.textContent = error;
  submit.disabled = false;
});

document.getElementById("cancel").addEventListener("click", () => window.close());
