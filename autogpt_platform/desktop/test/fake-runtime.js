"use strict";

// Stands in for the real runtime: reports progress, then ready, then waits
// for the shell to close stdin and exits cleanly.
const mode = process.argv[2] || "normal";

console.log(JSON.stringify({ event: "progress", step: "boot", message: "Booting" }));
console.log("a plain log line");
if (mode === "crash") {
  console.log(JSON.stringify({ event: "error", message: "boom", fatal: true }));
  process.exit(3);
}
console.log(JSON.stringify({ event: "ready", url: "http://127.0.0.1:43117" }));

if (mode === "ignore-stdin") {
  setInterval(() => {}, 1000);
} else if (mode === "slow-stop") {
  process.stdin.on("end", () => {
    console.log(JSON.stringify({ event: "progress", message: "Finishing", grace_seconds: 30 }));
    setTimeout(() => process.exit(0), 1500);
  });
  process.stdin.resume();
} else if (mode === "running") {
  // With an agent run in progress. Asked to stop it, it says how long that
  // may take and then takes a while ("running slow": for ever), unless
  // stdin closes first.
  console.log(JSON.stringify({ event: "activity", graph_runs: 1, copilot_turns: 0 }));
  const say = (message, more = {}) => console.log(JSON.stringify({ event: "progress", message, ...more }));
  require("node:readline")
    .createInterface({ input: process.stdin })
    .on("line", (line) => {
      say(`asked: ${line}`, { step: "stopping", grace_seconds: 30 });
      if (process.argv[3] !== "slow") setTimeout(() => (say("the run was stopped"), process.exit(0)), 1500);
    })
    .on("close", () => (say("stdin closed"), process.exit(0)));
} else {
  process.stdin.on("end", () => {
    console.log(JSON.stringify({ event: "progress", message: "stopped" }));
    process.exit(0);
  });
  process.stdin.resume();
}
