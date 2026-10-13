// Dedicated graph execution load testing
import http from "k6/http";
import { check, sleep, group } from "k6";
import { Rate, Trend, Counter } from "k6/metrics";
import { getEnvironmentConfig } from "../../configs/environment.js";
import { authHeaders, authSetupTimeout, setupAuth } from "../../lib/auth.js";
// Test data generation functions
function generateTestGraph(name = null) {
  const graphName =
    name || `Load Test Graph ${Math.random().toString(36).substr(2, 9)}`;
  return {
    name: graphName,
    description: "Generated graph for load testing purposes",
    graph: {
      name: graphName,
      description: "Load testing graph",
      nodes: [
        {
          id: "input_node",
          name: "Agent Input",
          block_id: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
          input_default: {
            name: "Load Test Input",
            description: "Test input for load testing",
          },
          input_nodes: [],
          output_nodes: ["output_node"],
          metadata: { position: { x: 100, y: 100 } },
        },
        {
          id: "output_node",
          name: "Agent Output",
          block_id: "363ae599-353e-4804-937e-b2ee3cef3da4",
          input_default: {
            name: "Load Test Output",
            description: "Test output for load testing",
            value: "Test output value",
          },
          input_nodes: ["input_node"],
          output_nodes: [],
          metadata: { position: { x: 300, y: 100 } },
        },
      ],
      links: [
        {
          source_id: "input_node",
          sink_id: "output_node",
          source_name: "result",
          sink_name: "value",
        },
      ],
    },
  };
}

// Graph inputs are keyed by the input block's `name`.
function generateExecutionInputs() {
  return {
    "Load Test Input": `Test execution at ${new Date().toISOString()}`,
  };
}

const config = getEnvironmentConfig();

// A 402 is a billing refusal: the account has no credits, or no subscription
// while the payment paywall applies to it. A new account can meet either,
// depending on the target's flags. The route answered as designed, so it is
// not a failed request, but no graph ran: graph_executions_total counts only
// the runs that started, and each refusal has its own counter.
const EXECUTION_STATUSES = http.expectedStatuses(200, 402);

// Custom metrics for graph execution testing
const graphCreations = new Counter("graph_creations_total");
const graphExecutions = new Counter("graph_executions_total");
const graphExecutionsNoCredits = new Counter(
  "graph_executions_refused_no_credits_total",
);
const graphExecutionsNoSubscription = new Counter(
  "graph_executions_refused_no_subscription_total",
);
const graphExecutionTime = new Trend("graph_execution_duration");
const graphCreationTime = new Trend("graph_creation_duration");
const executionErrors = new Rate("execution_errors");

// Which billing refusal a 402 is, from its detail; null for any other 402.
function billingRefusal(response) {
  let detail = "";
  try {
    detail = String(response.json("detail") || "");
  } catch (e) {
    detail = "";
  }
  if (/insufficient balance/i.test(detail)) return "no credits";
  if (/subscription is required/i.test(detail)) return "no subscription";
  return null;
}

// Configurable options for easy load adjustment
export const options = {
  stages: [
    { duration: __ENV.RAMP_UP || "1m", target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.DURATION || "5m", target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.RAMP_DOWN || "1m", target: 0 },
  ],
  // Thresholds disabled to prevent test abortion - collect all performance data
  // thresholds: {
  //   checks: ['rate>0.60'],
  //   http_req_duration: ['p(95)<45000', 'p(99)<60000'],
  //   http_req_failed: ['rate<0.4'],
  //   graph_execution_duration: ['p(95)<45000'],
  //   graph_creation_duration: ['p(95)<30000'],
  // },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: "AutoGPT Platform - Graph Creation & Execution Test",
  },
  // Timeout configurations to prevent early termination
  setupTimeout: authSetupTimeout(),
  teardownTimeout: "60s",
  noConnectionReuse: false,
  userAgent: "k6-load-test/1.0",
};

export function setup() {
  console.log("🎯 Setting up graph execution load test...");
  console.log(
    `Configuration: VUs=${parseInt(__ENV.VUS) || 5}, Duration=${__ENV.DURATION || "2m"}`,
  );
  return {
    ...setupAuth(),
    timestamp: Date.now(),
  };
}

export default function (data) {
  // Get load multiplier - how many concurrent operations each VU should perform
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;
  const headers = authHeaders(data);

  console.log(
    `🚀 VU ${__VU} performing ${requestsPerVU} concurrent graph operations...`,
  );

  // Create requests for concurrent execution
  const graphRequests = [];

  for (let i = 0; i < requestsPerVU; i++) {
    // Generate graph data
    const graphData = generateTestGraph();

    // Add graph creation request
    graphRequests.push({
      method: "POST",
      url: `${config.API_BASE_URL}/graphs`,
      body: JSON.stringify(graphData),
      params: { headers },
    });
  }

  // Execute all graph creations concurrently
  console.log(`📊 Creating ${requestsPerVU} graphs concurrently...`);
  const responses = http.batch(graphRequests);

  // Process results
  let successCount = 0;
  const createdGraphs = [];

  for (let i = 0; i < responses.length; i++) {
    const response = responses[i];

    const success = check(response, {
      [`Graph ${i + 1} created successfully`]: (r) => r.status === 200,
    });

    if (success && response.status === 200) {
      successCount++;
      try {
        const graph = JSON.parse(response.body);
        createdGraphs.push(graph);
        graphCreations.add(1);
      } catch (e) {
        console.error(`Error parsing graph ${i + 1} response:`, e);
      }
    } else {
      console.log(`❌ Graph ${i + 1} creation failed: ${response.status}`);
    }
  }

  console.log(
    `✅ VU ${__VU} created ${successCount}/${requestsPerVU} graphs concurrently`,
  );

  // Execute a subset of created graphs (to avoid overloading execution)
  const graphsToExecute = createdGraphs.slice(
    0,
    Math.min(5, createdGraphs.length),
  );

  if (graphsToExecute.length > 0) {
    console.log(`⚡ Executing ${graphsToExecute.length} graphs...`);

    const executionRequests = [];

    for (const graph of graphsToExecute) {
      const executionInputs = generateExecutionInputs();

      executionRequests.push({
        method: "POST",
        url: `${config.API_BASE_URL}/graphs/${graph.id}/execute/${graph.version}`,
        body: JSON.stringify({
          inputs: executionInputs,
          credentials_inputs: {},
        }),
        params: { headers, responseCallback: EXECUTION_STATUSES },
      });
    }

    // Execute graphs concurrently
    const executionResponses = http.batch(executionRequests);

    let executionSuccessCount = 0;
    const refusals = { "no credits": 0, "no subscription": 0 };
    for (let i = 0; i < executionResponses.length; i++) {
      const response = executionResponses[i];
      const refusal = response.status === 402 ? billingRefusal(response) : null;

      check(response, {
        [`Graph ${i + 1} execution started, or refused for no credits or no subscription (402)`]:
          (r) => r.status === 200 || refusal !== null,
      });

      if (response.status === 200) {
        executionSuccessCount++;
        graphExecutions.add(1);
      } else if (refusal === "no credits") {
        refusals[refusal]++;
        graphExecutionsNoCredits.add(1);
      } else if (refusal === "no subscription") {
        refusals[refusal]++;
        graphExecutionsNoSubscription.add(1);
      }
    }

    console.log(
      `✅ VU ${__VU} executed ${executionSuccessCount}/${graphsToExecute.length} graphs ` +
        `(refused: ${refusals["no credits"]} no credits, ${refusals["no subscription"]} no subscription)`,
    );
  }

  // Think time between iterations
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

export function teardown(data) {
  // graph_creations_total and graph_executions_total are in k6's summary.
  console.log("🧹 Cleaning up graph execution load test...");

  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}
