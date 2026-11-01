import http from "k6/http";
import { check, sleep } from "k6";
import { Counter } from "k6/metrics";
import {
  getEnvironmentConfig,
  PERFORMANCE_CONFIG,
} from "../../configs/environment.js";
import { authHeaders, authSetupTimeout, setupAuth } from "../../lib/auth.js";

const config = getEnvironmentConfig();

// Custom metrics
const userOperations = new Counter("user_operations_total");
const graphOperations = new Counter("graph_operations_total");
const executionOperations = new Counter("execution_operations_total");

// Test configuration for normal load testing
export const options = {
  stages: [
    {
      duration: __ENV.RAMP_UP || "1m",
      target: parseInt(__ENV.VUS) || PERFORMANCE_CONFIG.DEFAULT_VUS,
    },
    {
      duration: __ENV.DURATION || "5m",
      target: parseInt(__ENV.VUS) || PERFORMANCE_CONFIG.DEFAULT_VUS,
    },
    { duration: __ENV.RAMP_DOWN || "1m", target: 0 },
  ],
  // maxDuration: '15m', // Removed - not supported in k6 cloud
  thresholds: {
    checks: ["rate>0.50"], // Reduced for high concurrency complex operations
    http_req_duration: ["p(95)<60000", "p(99)<60000"], // Allow up to 60s response times
    http_req_failed: ["rate<0.5"], // Allow 50% failure rate for stress testing
  },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: "AutoGPT Platform - Full Platform Integration Test",
  },
  // Timeout configurations to prevent early termination
  setupTimeout: authSetupTimeout(),
  teardownTimeout: "60s",
  noConnectionReuse: false,
  userAgent: "k6-load-test/1.0",
};

export function setup() {
  console.log("🎯 Setting up load test scenario...");
  return {
    ...setupAuth(),
    timestamp: Date.now(),
  };
}

export default function (data) {
  // Get load multiplier - how many concurrent user journeys each VU should simulate
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;
  const headers = authHeaders(data);

  console.log(
    `🚀 VU ${__VU} simulating ${requestsPerVU} realistic user workflows...`,
  );

  // Create concurrent requests for all user journeys
  const requests = [];

  // Simulate realistic user workflows instead of just API hammering
  for (let i = 0; i < requestsPerVU; i++) {
    // Workflow 1: User checking their dashboard
    requests.push({
      method: "GET",
      url: `${config.API_BASE_URL}/credits`,
      params: { headers },
    });
    requests.push({
      method: "GET",
      url: `${config.API_BASE_URL}/graphs`,
      params: { headers },
    });

    // Workflow 2: User exploring available blocks for building agents
    requests.push({
      method: "GET",
      url: `${config.API_BASE_URL}/blocks`,
      params: { headers },
    });

    // Workflow 3: User monitoring their recent executions
    requests.push({
      method: "GET",
      url: `${config.API_BASE_URL}/executions`,
      params: { headers },
    });
  }

  console.log(
    `📊 Executing ${requests.length} requests across realistic user workflows...`,
  );

  // Execute all requests concurrently
  const responses = http.batch(requests);

  // Process results and count successes
  let creditsSuccesses = 0,
    graphsSuccesses = 0,
    blocksSuccesses = 0,
    executionsSuccesses = 0;

  for (let i = 0; i < responses.length; i++) {
    const response = responses[i];
    const operationType = i % 4; // Each set of 4 requests: 0=credits, 1=graphs, 2=blocks, 3=executions

    switch (operationType) {
      case 0: // Dashboard: Check credits
        if (
          check(response, {
            "Dashboard: User credits loaded successfully": (r) =>
              r.status === 200,
          })
        ) {
          creditsSuccesses++;
          userOperations.add(1);
        }
        break;
      case 1: // Dashboard: View graphs
        if (
          check(response, {
            "Dashboard: User graphs loaded successfully": (r) =>
              r.status === 200,
          })
        ) {
          graphsSuccesses++;
          graphOperations.add(1);
        }
        break;
      case 2: // Exploration: Browse available blocks
        if (
          check(response, {
            "Block Explorer: Available blocks loaded successfully": (r) =>
              r.status === 200,
          })
        ) {
          blocksSuccesses++;
          userOperations.add(1);
        }
        break;
      case 3: // Monitoring: Check execution history
        if (
          check(response, {
            "Execution Monitor: Recent executions loaded successfully": (r) =>
              r.status === 200,
          })
        ) {
          executionsSuccesses++;
          executionOperations.add(1);
        }
        break;
    }
  }

  console.log(
    `✅ VU ${__VU} completed realistic workflows: ${creditsSuccesses} dashboard checks, ${graphsSuccesses} graph views, ${blocksSuccesses} block explorations, ${executionsSuccesses} execution monitors`,
  );

  // Think time between user sessions
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

export function teardown(data) {
  console.log("🧹 Cleaning up load test...");
  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}
