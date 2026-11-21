import { defineConfig } from "@playwright/test";

// One installed app, one machine: the files run in name order in a single
// worker, each building on the one before. Nothing is retried; a retry would
// hide exactly the start, stop and leftover-process faults these tests exist
// to find.
//
// A failure ends its own file (the tests in a file are serial) and not the
// run: a later file whose prerequisites are missing says so (lib/state.ts
// loadState), and one whose prerequisites are there still has something to
// prove. The check at the end of 01 that is known to be red must not keep
// 02 to 04 from running.
export default defineConfig({
  testDir: "tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 20 * 60_000,
  expect: { timeout: 30_000 },
  outputDir: "test-results",
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  // No `use`: the tests attach to the installed app's window themselves
  // (lib/app.ts), which also records the trace that is kept on a failure.
});
