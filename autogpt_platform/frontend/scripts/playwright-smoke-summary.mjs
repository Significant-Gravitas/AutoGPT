import fs from "fs";
import path from "path";

const inputPath =
  process.env.PLAYWRIGHT_JSON_OUTPUT_FILE ||
  path.resolve(process.cwd(), "test-results", "smoke-report.json");
const outputDirectory = path.resolve(process.cwd(), "test-results");
const jsonOutputPath = path.join(outputDirectory, "smoke-summary.json");
const markdownOutputPath = path.join(outputDirectory, "smoke-summary.md");

function ensureOutputDirectory() {
  fs.mkdirSync(outputDirectory, { recursive: true });
}

function collectSpecs(suites = [], parentTitles = []) {
  const specs = [];

  for (const suite of suites) {
    const titles = suite.title ? [...parentTitles, suite.title] : parentTitles;

    for (const spec of suite.specs ?? []) {
      specs.push({
        file: spec.file,
        title: spec.title,
        fullTitle: [...titles, spec.title].filter(Boolean).join(" › "),
        tests: spec.tests ?? [],
      });
    }

    specs.push(...collectSpecs(suite.suites ?? [], titles));
  }

  return specs;
}

function getFinalStatus(test) {
  if (test.status) return test.status;
  const lastResult = test.results?.[test.results.length - 1];
  return lastResult?.status ?? "unknown";
}

function getRetries(test) {
  const attempts = (test.results ?? []).filter(
    (result) => result.status !== "skipped",
  );
  return Math.max(attempts.length - 1, 0);
}

function isFlaky(test) {
  const results = test.results ?? [];
  if (results.length < 2) return false;
  const finalStatus = getFinalStatus(test);
  return (
    finalStatus === "passed" &&
    results.slice(0, -1).some((result) => result.status !== "passed")
  );
}

function toFlowResult(spec) {
  const primaryTest = spec.tests[0];
  const status = primaryTest ? getFinalStatus(primaryTest) : "unknown";
  const retries = primaryTest ? getRetries(primaryTest) : 0;
  const flaky = primaryTest ? isFlaky(primaryTest) : false;

  return {
    flow: spec.title,
    fullTitle: spec.fullTitle,
    file: spec.file,
    status,
    retries,
    flaky,
  };
}

function buildSummary(results) {
  const passed = results.filter((result) => result.status === "passed").length;
  const failed = results.filter((result) =>
    ["failed", "timedOut", "interrupted"].includes(result.status),
  ).length;
  const skipped = results.filter(
    (result) => result.status === "skipped",
  ).length;
  const flaky = results.filter((result) => result.flaky).length;

  return {
    generatedAt: new Date().toISOString(),
    totals: {
      flows: results.length,
      passed,
      failed,
      skipped,
      flaky,
    },
    flows: results,
  };
}

function buildMarkdown(summary) {
  const lines = [
    "## Playwright PR Smoke",
    "",
    `- Total flows: ${summary.totals.flows}`,
    `- Passed: ${summary.totals.passed}`,
    `- Failed: ${summary.totals.failed}`,
    `- Skipped: ${summary.totals.skipped}`,
    `- Flaky after retry: ${summary.totals.flaky}`,
    "",
    "| Flow | Status | Retries | File |",
    "| --- | --- | --- | --- |",
  ];

  for (const flow of summary.flows) {
    lines.push(
      `| ${flow.flow} | ${flow.flaky ? "flaky-pass" : flow.status} | ${flow.retries} | ${flow.file ?? "n/a"} |`,
    );
  }

  if (summary.totals.failed > 0) {
    lines.push("");
    lines.push(
      "Failure artifacts, screenshots, and traces are available in the uploaded Playwright artifacts for this workflow run.",
    );
  }

  return `${lines.join("\n")}\n`;
}

ensureOutputDirectory();

if (!fs.existsSync(inputPath)) {
  throw new Error(`Playwright smoke report not found at ${inputPath}`);
}

const rawReport = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const specs = collectSpecs(rawReport.suites ?? []);
const smokeSpecs = specs.filter((spec) => spec.title.includes("@smoke"));
const summary = buildSummary(smokeSpecs.map(toFlowResult));

fs.writeFileSync(jsonOutputPath, JSON.stringify(summary, null, 2));
fs.writeFileSync(markdownOutputPath, buildMarkdown(summary));

console.log(`Smoke summary written to ${jsonOutputPath}`);
console.log(`Smoke markdown written to ${markdownOutputPath}`);
