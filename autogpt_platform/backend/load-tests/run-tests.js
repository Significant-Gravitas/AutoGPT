#!/usr/bin/env node
/**
 * Unified Load Test Runner
 *
 * Supports both local execution and k6 cloud execution with the same interface.
 * Automatically detects cloud credentials and provides seamless switching.
 *
 * Usage:
 *   node run-tests.js verify                     # Quick verification (1 VU, 10s)
 *   node run-tests.js run core-api-test DEV      # Run specific test locally
 *   node run-tests.js run all DEV                # Run all tests locally
 *   node run-tests.js cloud core-api DEV         # Run specific test in k6 cloud
 *   node run-tests.js cloud all DEV              # Run all tests in k6 cloud
 */

import { execSync } from "child_process";
import fs from "fs";

const TESTS = {
  "connectivity-test": {
    script: "tests/basic/connectivity-test.js",
    description: "Basic connectivity validation",
    cloudConfig: { vus: 10, duration: "2m" },
  },
  "single-endpoint-test": {
    script: "tests/basic/single-endpoint-test.js",
    description: "Individual API endpoint testing",
    cloudConfig: { vus: 25, duration: "3m" },
  },
  "core-api-test": {
    script: "tests/api/core-api-test.js",
    description: "Core API endpoints performance test",
    cloudConfig: { vus: 100, duration: "5m" },
  },
  "graph-execution-test": {
    script: "tests/api/graph-execution-test.js",
    description: "Graph creation and execution pipeline test",
    cloudConfig: { vus: 80, duration: "5m" },
  },
  "marketplace-public-test": {
    script: "tests/marketplace/public-access-test.js",
    description: "Public marketplace browsing test",
    cloudConfig: { vus: 150, duration: "3m" },
  },
  "marketplace-library-test": {
    script: "tests/marketplace/library-access-test.js",
    description: "Authenticated marketplace/library test",
    cloudConfig: { vus: 100, duration: "4m" },
  },
  "comprehensive-test": {
    script: "tests/comprehensive/platform-journey-test.js",
    description: "Complete user journey simulation",
    cloudConfig: { vus: 50, duration: "6m" },
  },
};

function checkCloudCredentials() {
  const token = process.env.K6_CLOUD_TOKEN;
  const projectId = process.env.K6_CLOUD_PROJECT_ID;

  if (!token || !projectId) {
    console.log("❌ Missing k6 cloud credentials");
    console.log("Set: K6_CLOUD_TOKEN and K6_CLOUD_PROJECT_ID");
    return false;
  }
  return true;
}

function verifySetup() {
  console.log("🔍 Quick Setup Verification");

  // Check tokens
  if (!fs.existsSync("configs/pre-authenticated-tokens.js")) {
    console.log("❌ No tokens found. Run: node generate-tokens.js");
    return false;
  }

  // Quick test
  try {
    execSync(
      "K6_ENVIRONMENT=DEV VUS=1 DURATION=10s k6 run tests/basic/connectivity-test.js --quiet",
      { stdio: "inherit", cwd: process.cwd() },
    );
    console.log("✅ Verification successful");
    return true;
  } catch (error) {
    console.log("❌ Verification failed");
    return false;
  }
}

function runLocalTest(testName, environment) {
  const test = TESTS[testName];
  if (!test) {
    console.log(`❌ Unknown test: ${testName}`);
    console.log("Available tests:", Object.keys(TESTS).join(", "));
    return;
  }

  console.log(`🚀 Running ${test.description} locally on ${environment}`);

  try {
    const cmd = `K6_ENVIRONMENT=${environment} VUS=5 DURATION=30s k6 run ${test.script}`;
    execSync(cmd, { stdio: "inherit", cwd: process.cwd() });
    console.log("✅ Test completed");
  } catch (error) {
    console.log("❌ Test failed");
  }
}

function runCloudTest(testName, environment) {
  const test = TESTS[testName];
  if (!test) {
    console.log(`❌ Unknown test: ${testName}`);
    console.log("Available tests:", Object.keys(TESTS).join(", "));
    return;
  }

  const { vus, duration } = test.cloudConfig;
  console.log(`☁️ Running ${test.description} in k6 cloud`);
  console.log(`   Environment: ${environment}`);
  console.log(`   Config: ${vus} VUs × ${duration}`);

  try {
    const cmd = `k6 cloud run --env K6_ENVIRONMENT=${environment} --env VUS=${vus} --env DURATION=${duration} --env RAMP_UP=30s --env RAMP_DOWN=30s ${test.script}`;
    const output = execSync(cmd, {
      stdio: "pipe",
      cwd: process.cwd(),
      encoding: "utf8",
    });

    // Extract and display URL
    const urlMatch = output.match(/https:\/\/[^\s]*grafana[^\s]*/);
    if (urlMatch) {
      const url = urlMatch[0];
      console.log(`🔗 Test URL: ${url}`);

      // Save to results file
      const timestamp = new Date().toISOString();
      const result = `${timestamp} - ${testName}: ${url}\n`;
      fs.appendFileSync("k6-cloud-results.txt", result);
    }

    console.log("✅ Cloud test started successfully");
  } catch (error) {
    console.log("❌ Cloud test failed to start");
    console.log(error.message);
  }
}

function runAllLocalTests(environment) {
  console.log(`🚀 Running all tests locally on ${environment}`);

  for (const [testName, test] of Object.entries(TESTS)) {
    console.log(`\n📊 ${test.description}`);
    runLocalTest(testName, environment);
  }
}

function runAllCloudTests(environment) {
  console.log(`☁️ Running all tests in k6 cloud on ${environment}`);

  const testNames = Object.keys(TESTS);
  for (let i = 0; i < testNames.length; i++) {
    const testName = testNames[i];
    console.log(`\n📊 Test ${i + 1}/${testNames.length}: ${testName}`);

    runCloudTest(testName, environment);

    // Brief pause between cloud tests (except last one)
    if (i < testNames.length - 1) {
      console.log("⏸️ Waiting 2 minutes before next cloud test...");
      execSync("sleep 120");
    }
  }
}

function listTests() {
  console.log("📋 Available Tests:");
  console.log("==================");

  Object.entries(TESTS).forEach(([name, test]) => {
    const { vus, duration } = test.cloudConfig;
    console.log(`  ${name.padEnd(20)} - ${test.description}`);
    console.log(`  ${" ".repeat(20)}   Cloud: ${vus} VUs × ${duration}`);
  });

  console.log("\n🌍 Available Environments: LOCAL, DEV, PROD");
  console.log("\n💡 Examples:");
  console.log("  # Local execution (5 VUs, 30s)");
  console.log("  node run-tests.js verify");
  console.log("  node run-tests.js run core-api-test DEV");
  console.log("  node run-tests.js run core-api-test,marketplace-test DEV");
  console.log("  node run-tests.js run all DEV");
  console.log("");
  console.log("  # Cloud execution (high VUs, longer duration)");
  console.log("  node run-tests.js cloud core-api DEV");
  console.log("  node run-tests.js cloud all DEV");

  const hasCloudCreds = checkCloudCredentials();
  console.log(
    `\n☁️ Cloud Status: ${hasCloudCreds ? "✅ Configured" : "❌ Missing credentials"}`,
  );
}

function runSequentialTests(testNames, environment, isCloud = false) {
  const tests = testNames.split(",").map((t) => t.trim());
  const mode = isCloud ? "cloud" : "local";
  console.log(
    `🚀 Running ${tests.length} tests sequentially in ${mode} mode on ${environment}`,
  );

  for (let i = 0; i < tests.length; i++) {
    const testName = tests[i];
    console.log(`\n📊 Test ${i + 1}/${tests.length}: ${testName}`);

    if (isCloud) {
      runCloudTest(testName, environment);
    } else {
      runLocalTest(testName, environment);
    }

    // Brief pause between tests (except last one)
    if (i < tests.length - 1) {
      const pauseTime = isCloud ? "2 minutes" : "10 seconds";
      const pauseCmd = isCloud ? "sleep 120" : "sleep 10";
      console.log(`⏸️ Waiting ${pauseTime} before next test...`);
      if (!isCloud) {
        // Note: In real implementation, would use setTimeout/sleep for local tests
      }
    }
  }
}

// Main CLI
const [, , command, testOrEnv, environment] = process.argv;

switch (command) {
  case "verify":
    verifySetup();
    break;
  case "list":
    listTests();
    break;
  case "run":
    if (testOrEnv === "all") {
      runAllLocalTests(environment || "DEV");
    } else if (testOrEnv?.includes(",")) {
      runSequentialTests(testOrEnv, environment || "DEV", false);
    } else {
      runLocalTest(testOrEnv, environment || "DEV");
    }
    break;
  case "cloud":
    if (!checkCloudCredentials()) {
      process.exit(1);
    }
    if (testOrEnv === "all") {
      runAllCloudTests(environment || "DEV");
    } else if (testOrEnv?.includes(",")) {
      runSequentialTests(testOrEnv, environment || "DEV", true);
    } else {
      runCloudTest(testOrEnv, environment || "DEV");
    }
    break;
  default:
    listTests();
}
