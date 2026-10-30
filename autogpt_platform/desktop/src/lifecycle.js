"use strict";

// Which runtime the shell listens to, without Electron so it can be tested.
//
// A restart stops one runtime and starts another, and the one being stopped
// can still speak: a `ready` it printed just as it was told to stop would
// open the window on a proxy that is shutting down, and a failure on its way
// out would be shown as the new start's. So only the current runtime is
// heard, and while it is being stopped only its request for more time
// ("Finishing a database update", see runtime.js) is passed on.
function hearsRuntime({ current, sender, restarting, event }) {
  if (sender !== current) return false;
  return !restarting || event?.step === "stopping";
}

module.exports = { hearsRuntime };
