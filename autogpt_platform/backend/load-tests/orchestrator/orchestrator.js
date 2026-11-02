#!/usr/bin/env node

/**
 * AutoGPT Platform load test orchestrator.
 *
 * Runs every scenario below in turn, each at its highest VU count: the full
 * stress suite, about an hour end to end.
 *
 *   node orchestrator/orchestrator.js [PREVIEW|DEV|LOCAL|PROD] [local|cloud]
 *
 * local (the default) runs k6 on this machine through ../k6.sh, which uses
 * Docker, or through the binary named by K6_BIN. cloud runs each test on
 * Grafana Cloud k6, which bills every virtual-user hour: it is only for load
 * one machine cannot generate, agreed with the team first, and it needs the
 * explicit `cloud` argument plus K6_CLOUD_TOKEN and K6_CLOUD_STACK_ID.
 */

import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const LOAD_TESTS_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const K6_BIN = process.env.K6_BIN || path.join(LOAD_TESTS_DIR, "k6.sh");
// Not a secret: the AutoGPT Platform project in Grafana Cloud k6.
const K6_CLOUD_PROJECT_ID = process.env.K6_CLOUD_PROJECT_ID || "4254406";
const PAUSE_BETWEEN_TESTS = 30; // seconds

const args = process.argv.slice(2);
const environment = (
  args[0] ||
  process.env.K6_ENVIRONMENT ||
  "DEV"
).toUpperCase();
const executionMode = (args[1] || "local").toLowerCase();

// Unified test scenarios - only highest VUs for each unique test
const unifiedTestScenarios = [
  // 1. Marketplace Public Access (highest VUs: 314)
  {
    name: "Marketplace_Public_Access_Max_Load",
    file: "tests/marketplace/public-access-test.js",
    vus: 314,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    description: "Public marketplace browsing at maximum load",
  },

  // 2. Marketplace Authenticated Access (highest VUs: 157)
  {
    name: "Marketplace_Authenticated_Access_Max_Load",
    file: "tests/marketplace/library-access-test.js",
    vus: 157,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    description: "Authenticated marketplace/library operations at maximum load",
  },

  // 3. Core API Load Test (highest VUs: 100)
  {
    name: "Core_API_Max_Load",
    file: "tests/api/core-api-test.js",
    vus: 100,
    duration: "5m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Core authenticated API endpoints at maximum load",
  },

  // 4. Graph Execution Load Test (highest VUs: 100)
  {
    name: "Graph_Execution_Max_Load",
    file: "tests/api/graph-execution-test.js",
    vus: 100,
    duration: "5m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Graph workflow execution pipeline at maximum load",
  },

  // 5. Credits API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Credits_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "credits", CONCURRENT_REQUESTS: "1" },
    description: "Credits API endpoint at maximum load",
  },

  // 6. Graphs API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Graphs_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "graphs", CONCURRENT_REQUESTS: "1" },
    description: "Graphs API endpoint at maximum load",
  },

  // 7. Blocks API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Blocks_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "blocks", CONCURRENT_REQUESTS: "1" },
    description: "Blocks API endpoint at maximum load",
  },

  // 8. Executions API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Executions_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "executions", CONCURRENT_REQUESTS: "1" },
    description: "Executions API endpoint at maximum load",
  },

  // 9. Comprehensive Platform Journey (highest VUs: 100)
  {
    name: "Comprehensive_Platform_Max_Load",
    file: "tests/comprehensive/platform-journey-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    description: "End-to-end user journey simulation at maximum load",
  },

  // 10. Marketplace Stress Test (highest VUs: 500)
  {
    name: "Marketplace_Stress_Test",
    file: "tests/marketplace/public-access-test.js",
    vus: 500,
    duration: "2m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Ultimate marketplace stress test",
  },

  // 11. Core API Stress Test (highest VUs: 500)
  {
    name: "Core_API_Stress_Test",
    file: "tests/api/core-api-test.js",
    vus: 500,
    duration: "2m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Ultimate core API stress test",
  },

  // 12. Long Duration Core API Test (highest VUs: 100, longest duration)
  {
    name: "Long_Duration_Core_API_Test",
    file: "tests/api/core-api-test.js",
    vus: 100,
    duration: "10m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Extended duration core API endurance test",
  },
];

// Problems that would fail every test, reported before the first one starts.
function findSetupProblems() {
  const problems = [];
  if (!["PREVIEW", "DEV", "LOCAL", "PROD"].includes(environment)) {
    problems.push(
      `Unknown environment "${environment}": use PREVIEW, DEV, LOCAL or PROD`,
    );
  }
  if (!["local", "cloud"].includes(executionMode)) {
    problems.push(`Unknown mode "${executionMode}": use local or cloud`);
  }
  // Streaming results to Grafana Cloud is billed like a cloud run, and a
  // K6_BIN binary would not go through k6.sh's opt-in check.
  if (executionMode === "local" && /cloud/i.test(process.env.K6_OUT || "")) {
    problems.push(
      "local mode does not stream results to Grafana Cloud, which bills every virtual-user hour: unset K6_OUT, or use cloud mode for an agreed cloud run",
    );
  }
  if (
    environment === "PREVIEW" &&
    !/^[1-9]\d{0,6}$/.test(process.env.PR_NUMBER || "")
  ) {
    problems.push("PREVIEW needs PR_NUMBER=<pull request number>");
  }
  // PREVIEW and LOCAL sign up throwaway accounts when this is unset.
  if (
    (environment === "DEV" || environment === "PROD") &&
    !process.env.LOAD_TEST_ACCOUNTS
  ) {
    problems.push(
      `${environment} needs LOAD_TEST_ACCOUNTS for the authenticated tests; see README.md`,
    );
  }
  if (executionMode === "cloud") {
    if (!process.env.K6_CLOUD_TOKEN) {
      problems.push(
        "cloud mode needs K6_CLOUD_TOKEN, a Grafana Cloud k6 API token",
      );
    }
    if (!process.env.K6_CLOUD_STACK_ID) {
      problems.push(
        "cloud mode needs K6_CLOUD_STACK_ID, the Grafana Cloud stack's numeric ID",
      );
    }
    if (process.env.LOAD_TEST_ACCOUNTS) {
      problems.push(
        "cloud mode does not send LOAD_TEST_ACCOUNTS, because Grafana Cloud stores a remote run's --env values with the run. " +
          "Use local mode, or on a preview unset LOAD_TEST_ACCOUNTS so the tests sign up throwaway accounts.",
      );
    }
  }
  return problems;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function minutes(duration) {
  const unit = { h: 60, m: 1, s: 1 / 60 };
  let total = 0;
  for (const [, value, suffix] of duration.matchAll(
    /(\d+(?:\.\d+)?)(h|m|s)/g,
  )) {
    total += Number(value) * unit[suffix];
  }
  return total;
}

// Grafana Cloud k6 bills each test as its peak VUs times its whole duration.
function suiteVirtualUserHours() {
  const vuMinutes = unifiedTestScenarios.reduce(
    (sum, test) =>
      sum +
      test.vus *
        (minutes(test.rampUp) +
          minutes(test.duration) +
          minutes(test.rampDown)),
    0,
  );
  return vuMinutes / 60;
}

function runTest(test, index) {
  return new Promise((resolve, reject) => {
    console.log(
      `\nTest ${index + 1}/${unifiedTestScenarios.length}: ${test.name}`,
    );
    console.log(
      `Config: ${test.vus} VUs x ${test.duration} (${executionMode} mode)`,
    );
    console.log(`Script: ${test.file}`);
    console.log(`Description: ${test.description}`);
    console.log(`Started: ${new Date().toISOString()}`);

    const testEnv = {
      K6_ENVIRONMENT: environment,
      K6_CLOUD_PROJECT_ID,
      VUS: test.vus.toString(),
      DURATION: test.duration,
      RAMP_UP: test.rampUp,
      RAMP_DOWN: test.rampDown,
      ...(process.env.PR_NUMBER ? { PR_NUMBER: process.env.PR_NUMBER } : {}),
      ...(process.env.LOAD_TEST_ACCOUNT_COUNT
        ? { LOAD_TEST_ACCOUNT_COUNT: process.env.LOAD_TEST_ACCOUNT_COUNT }
        : {}),
      ...test.env,
    };

    // A remote cloud run sees only the variables passed with --env. A local
    // run reads them, and any secrets, from the environment k6 inherits, so
    // nothing secret ever goes on a command line.
    const k6Args =
      executionMode === "cloud"
        ? [
            "cloud",
            "run",
            ...Object.entries(testEnv).flatMap(([key, value]) => [
              "--env",
              `${key}=${value}`,
            ]),
            test.file,
          ]
        : ["run", test.file];
    // k6.sh refuses Grafana Cloud unless its first argument is --cloud.
    const wrapperArgs =
      executionMode === "cloud" && !process.env.K6_BIN ? ["--cloud"] : [];

    const k6Process = spawn(K6_BIN, [...wrapperArgs, ...k6Args], {
      cwd: LOAD_TESTS_DIR,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...testEnv },
    });

    let runUrl = null;

    k6Process.stdout.on("data", (data) => {
      const str = data.toString();

      const runUrlMatch = str.match(/https?:\/\/\S+?\/runs\/\d+/);
      if (runUrlMatch && !runUrl) {
        runUrl = runUrlMatch[0];
        console.log(`Run URL: ${runUrl}`);
      }

      const progressMatch = str.match(/(\d+)%/);
      if (progressMatch) {
        process.stdout.write(`\rProgress: ${progressMatch[1]}%`);
      }
    });

    let stderrTail = "";
    k6Process.stderr.on("data", (data) => {
      stderrTail = (stderrTail + data.toString()).slice(-2000);
    });

    k6Process.on("close", (code) => {
      process.stdout.write("\n");
      const result = {
        success: code === 0,
        url: runUrl || "unknown",
        vus: test.vus,
        duration: test.duration,
      };
      if (code === 0) {
        console.log(`${test.name} SUCCESS`);
      } else {
        console.log(`${test.name} FAILED (exit code ${code})`);
        if (stderrTail) console.log(stderrTail.trim());
        result.exitCode = code;
      }
      resolve(result);
    });

    k6Process.on("error", (error) => {
      console.log(`${test.name} ERROR: ${error.message}`);
      reject(error);
    });
  });
}

async function main() {
  console.log("AUTOGPT PLATFORM LOAD TEST ORCHESTRATOR\n");
  console.log(`Target environment: ${environment}`);
  console.log(`Execution mode: ${executionMode}`);

  const problems = findSetupProblems();
  if (problems.length > 0) {
    for (const problem of problems) console.error(`Error: ${problem}`);
    process.exit(2);
  }
  if (executionMode === "cloud") {
    console.log(
      `Grafana Cloud k6 bills every virtual-user hour, and this suite is about ${Math.round(suiteVirtualUserHours())} VUh a run; local mode is the default.`,
    );
  }

  console.log(`\nUNIFIED TEST PLAN`);
  console.log(`Total tests: ${unifiedTestScenarios.length}`);
  console.log(`Estimated duration: ~60 minutes\n`);
  unifiedTestScenarios.forEach((test, i) => {
    console.log(
      `  ${i + 1}. ${test.name} (${test.vus} VUs x ${test.duration})`,
    );
  });

  const results = [];

  for (let i = 0; i < unifiedTestScenarios.length; i++) {
    const test = unifiedTestScenarios[i];

    try {
      const result = await runTest(test, i);
      results.push({ ...test, ...result });

      if (i < unifiedTestScenarios.length - 1) {
        console.log(
          `\nPausing ${PAUSE_BETWEEN_TESTS}s before the next test...`,
        );
        await sleep(PAUSE_BETWEEN_TESTS * 1000);
      }
    } catch (error) {
      console.error(`Fatal error running ${test.name}:`, error.message);
      results.push({ ...test, success: false, error: error.message });
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("UNIFIED LOAD TEST RESULTS SUMMARY");
  console.log("=".repeat(60));

  const successful = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  console.log(
    `Successful tests: ${successful.length}/${results.length} (${Math.round((successful.length / results.length) * 100)}%)`,
  );
  console.log(`Failed tests: ${failed.length}/${results.length}`);

  if (successful.length > 0) {
    console.log("\nSUCCESSFUL TESTS:");
    successful.forEach((test) => {
      console.log(`  - ${test.name} (${test.vus} VUs) - ${test.url}`);
    });
  }

  if (failed.length > 0) {
    console.log("\nFAILED TESTS:");
    failed.forEach((test) => {
      console.log(
        `  - ${test.name} (${test.vus} VUs) - ${test.url || "no URL"} (exit: ${test.exitCode || "unknown"})`,
      );
    });
  }

  const resultsDir = path.join(LOAD_TESTS_DIR, "results");
  fs.mkdirSync(resultsDir, { recursive: true });
  const resultsFile = path.join(
    resultsDir,
    `unified-results-${Math.floor(Date.now() / 1000)}.json`,
  );
  fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));
  console.log(`\nDetailed results saved to: ${resultsFile}`);

  process.exit(failed.length === 0 ? 0 : 1);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}
