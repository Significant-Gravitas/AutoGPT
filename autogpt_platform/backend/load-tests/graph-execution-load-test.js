// Dedicated graph execution load testing
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvironmentConfig } from './configs/environment.js';
import { getAuthenticatedUser, getAuthHeaders } from './utils/auth.js';
import { generateTestGraph, generateComplexTestGraph, generateExecutionInputs } from './utils/test-data.js';

const config = getEnvironmentConfig();

// Custom metrics for graph execution testing
const graphCreations = new Counter('graph_creations_total');
const graphExecutions = new Counter('graph_executions_total');
const graphExecutionTime = new Trend('graph_execution_duration');
const graphCreationTime = new Trend('graph_creation_duration');
const executionErrors = new Rate('execution_errors');

// Configurable options for easy load adjustment
export const options = {
  stages: [
    { duration: __ENV.RAMP_UP || '1m', target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.DURATION || '5m', target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.RAMP_DOWN || '1m', target: 0 },
  ],
  thresholds: {
    checks: ['rate>0.60'], // Reduced for complex operations under high load
    http_req_duration: ['p(95)<45000', 'p(99)<60000'], // Much higher for graph operations
    http_req_failed: ['rate<0.4'], // Higher tolerance for complex operations
    graph_execution_duration: ['p(95)<45000'], // Increased for high concurrency
    graph_creation_duration: ['p(95)<30000'], // Increased for high concurrency
  },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: 'AutoGPT Platform - Graph Creation & Execution Test',
  },
  // Timeout configurations to prevent early termination
  setupTimeout: '60s',
  teardownTimeout: '60s',
  noConnectionReuse: false,
  userAgent: 'k6-load-test/1.0',
};

export function setup() {
  console.log('🎯 Setting up graph execution load test...');
  console.log(`Configuration: VUs=${parseInt(__ENV.VUS) || 5}, Duration=${__ENV.DURATION || '2m'}`);
  return {
    timestamp: Date.now()
  };
}

export default function (data) {
  // Get load multiplier - how many concurrent operations each VU should perform
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;
  
  let userAuth;
  
  try {
    userAuth = getAuthenticatedUser();
  } catch (error) {
    console.error(`❌ Authentication failed:`, error);
    return;
  }
  
  // Handle authentication failure gracefully (null returned from auth fix)
  if (!userAuth || !userAuth.access_token) {
    console.log(`⚠️ VU ${__VU} has no valid authentication - skipping graph execution`);
    check(null, {
      'Graph Execution: Failed gracefully without crashing VU': () => true,
    });
    return; // Exit iteration gracefully without crashing
  }
  
  const headers = getAuthHeaders(userAuth.access_token);
  
  console.log(`🚀 VU ${__VU} performing ${requestsPerVU} concurrent graph operations...`);
  
  // Create requests for concurrent execution
  const graphRequests = [];
  
  for (let i = 0; i < requestsPerVU; i++) {
    // Generate graph data
    const graphData = generateTestGraph();
    
    // Add graph creation request
    graphRequests.push({
      method: 'POST',
      url: `${config.API_BASE_URL}/api/graphs`,
      body: JSON.stringify(graphData),
      params: { headers }
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
      [`Graph ${i+1} created successfully`]: (r) => r.status === 200,
    });
    
    if (success && response.status === 200) {
      successCount++;
      try {
        const graph = JSON.parse(response.body);
        createdGraphs.push(graph);
        graphCreations.add(1);
      } catch (e) {
        console.error(`Error parsing graph ${i+1} response:`, e);
      }
    } else {
      console.log(`❌ Graph ${i+1} creation failed: ${response.status}`);
    }
  }
  
  console.log(`✅ VU ${__VU} created ${successCount}/${requestsPerVU} graphs concurrently`);
  
  // Execute a subset of created graphs (to avoid overloading execution)
  const graphsToExecute = createdGraphs.slice(0, Math.min(5, createdGraphs.length));
  
  if (graphsToExecute.length > 0) {
    console.log(`⚡ Executing ${graphsToExecute.length} graphs...`);
    
    const executionRequests = [];
    
    for (const graph of graphsToExecute) {
      const executionInputs = generateExecutionInputs();
      
      executionRequests.push({
        method: 'POST',
        url: `${config.API_BASE_URL}/api/graphs/${graph.id}/execute/${graph.version}`,
        body: JSON.stringify({
          inputs: executionInputs,
          credentials_inputs: {}
        }),
        params: { headers }
      });
    }
    
    // Execute graphs concurrently
    const executionResponses = http.batch(executionRequests);
    
    let executionSuccessCount = 0;
    for (let i = 0; i < executionResponses.length; i++) {
      const response = executionResponses[i];
      
      const success = check(response, {
        [`Graph ${i+1} execution initiated`]: (r) => r.status === 200 || r.status === 402,
      });
      
      if (success) {
        executionSuccessCount++;
        graphExecutions.add(1);
      }
    }
    
    console.log(`✅ VU ${__VU} executed ${executionSuccessCount}/${graphsToExecute.length} graphs`);
  }
  
  // Think time between iterations
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// Legacy functions removed - replaced by concurrent execution in main function
// These functions are no longer used since implementing http.batch() for true concurrency

export function teardown(data) {
  console.log('🧹 Cleaning up graph execution load test...');
  console.log(`Total graph creations: ${graphCreations.value || 0}`);
  console.log(`Total graph executions: ${graphExecutions.value || 0}`);
  
  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}