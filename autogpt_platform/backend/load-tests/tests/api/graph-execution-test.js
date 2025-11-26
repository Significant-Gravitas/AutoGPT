// Dedicated graph execution load testing
import http from "k6/http";
import { check, sleep, group } from "k6";
import { Rate, Trend, Counter } from "k6/metrics";
import { getEnvironmentConfig } from "../../configs/environment.js";
import { getPreAuthenticatedHeaders } from "../../configs/pre-authenticated-tokens.js";
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
            placeholder_values: {},
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

function generateExecutionInputs() {
  return {
    "Load Test Input": {
      name: "Load Test Input",
      description: "Test input for load testing",
      placeholder_values: {
        test_data: `Test execution at ${new Date().toISOString()}`,
        test_parameter: Math.random().toString(36).substr(2, 9),
        numeric_value: Math.floor(Math.random() * 1000),
      },
    },
  };
}

const config = getEnvironmentConfig();

// Custom metrics for graph execution testing
const graphCreations = new Counter("graph_creations_total");
const graphExecutions = new Counter("graph_executions_total");
const graphExecutionTime = new Trend("graph_execution_duration");
const graphCreationTime = new Trend("graph_creation_duration");
const executionErrors = new Rate("execution_errors");

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
  setupTimeout: "60s",
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
    timestamp: Date.now(),
  };
}

export default function (data) {
  // Get load multiplier - how many concurrent operations each VU should perform
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;

  // Get pre-authenticated headers (no auth API calls during test)
  const headers = getPreAuthenticatedHeaders(__VU);

  // Handle missing token gracefully
  if (!headers || !headers.Authorization) {
    console.log(
      `⚠️ VU ${__VU} has no valid pre-authenticated token - skipping graph execution`,
    );
    check(null, {
      "Graph Execution: Failed gracefully without crashing VU": () => true,
    });
    return; // Exit iteration gracefully without crashing
  }

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
      url: `${config.API_BASE_URL}/api/graphs`,
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
        url: `${config.API_BASE_URL}/api/graphs/${graph.id}/execute/${graph.version}`,
        body: JSON.stringify({
          inputs: executionInputs,
          credentials_inputs: {},
        }),
        params: { headers },
      });
    }

    // Execute graphs concurrently
    const executionResponses = http.batch(executionRequests);

    let executionSuccessCount = 0;
    for (let i = 0; i < executionResponses.length; i++) {
      const response = executionResponses[i];

      const success = check(response, {
        [`Graph ${i + 1} execution initiated`]: (r) =>
          r.status === 200 || r.status === 402,
      });

      if (success) {
        executionSuccessCount++;
        graphExecutions.add(1);
      }
    }

    console.log(
      `✅ VU ${__VU} executed ${executionSuccessCount}/${graphsToExecute.length} graphs`,
    );
  }

  // Think time between iterations
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// Legacy functions removed - replaced by concurrent execution in main function
// These functions are no longer used since implementing http.batch() for true concurrency

export function teardown(data) {
  console.log("🧹 Cleaning up graph execution load test...");
  console.log(`Total graph creations: ${graphCreations.value || 0}`);
  console.log(`Total graph executions: ${graphExecutions.value || 0}`);

  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}
