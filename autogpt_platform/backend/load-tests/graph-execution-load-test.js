// Dedicated graph execution load testing
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';
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
    { duration: __ENV.RAMP_UP || '30s', target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.DURATION || '2m', target: parseInt(__ENV.VUS) || 5 },
    { duration: __ENV.RAMP_DOWN || '30s', target: 0 },
  ],
  thresholds: {
    checks: ['rate>0.95'],
    http_req_duration: ['p(95)<5000', 'p(99)<10000'],
    http_req_failed: ['rate<0.1'],
    graph_execution_duration: ['p(95)<10000'],
    graph_creation_duration: ['p(95)<3000'],
    execution_errors: ['rate<0.1'],
  },
  ext: {
    loadimpact: {
      projectID: __ENV.K6_CLOUD_PROJECT_ID,
      name: 'AutoGPT Graph Execution Load Test',
    },
  },
};

export function setup() {
  console.log('🎯 Setting up graph execution load test...');
  console.log(`Configuration: VUs=${parseInt(__ENV.VUS) || 5}, Duration=${__ENV.DURATION || '2m'}`);
  return {
    timestamp: Date.now()
  };
}

export default function (data) {
  const testUser = getRandomTestUser();
  let userAuth;
  
  try {
    userAuth = authenticateUser(testUser);
  } catch (error) {
    console.error(`❌ Authentication failed for ${testUser.email}:`, error);
    return;
  }
  
  const headers = getAuthHeaders(userAuth.access_token);
  
  // Graph Creation and Execution Journey
  group('Graph Creation and Execution Flow', () => {
    graphCreationAndExecutionJourney(headers);
  });
  
  // Complex Graph Testing
  group('Complex Graph Execution', () => {
    complexGraphExecutionJourney(headers);
  });
  
  // Graph Management Operations
  group('Graph Management', () => {
    graphManagementJourney(headers);
  });
  
  // Think time between iterations
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

function graphCreationAndExecutionJourney(headers) {
  const startTime = Date.now();
  
  // 1. Create a simple test graph
  console.log('🔨 Creating test graph...');
  const graphData = generateTestGraph();
  
  const createStartTime = Date.now();
  const createResponse = http.post(
    `${config.API_BASE_URL}/api/graphs`,
    JSON.stringify(graphData),
    { headers }
  );
  
  graphCreationTime.add(Date.now() - createStartTime);
  graphCreations.add(1);
  
  const createSuccess = check(createResponse, {
    'Graph created successfully': (r) => r.status === 200,
    'Graph response has ID': (r) => {
      try {
        const graph = JSON.parse(r.body);
        return graph.id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (createSuccess && createResponse.status === 200) {
    try {
      const createdGraph = JSON.parse(createResponse.body);
      console.log(`✅ Graph created: ${createdGraph.id}`);
      
      // 2. Execute the graph
      console.log('⚡ Executing graph...');
      const executionInputs = generateExecutionInputs();
      
      const executeStartTime = Date.now();
      const executeResponse = http.post(
        `${config.API_BASE_URL}/api/graphs/${createdGraph.id}/execute/${createdGraph.version}`,
        JSON.stringify({
          inputs: executionInputs,
          credentials_inputs: {}
        }),
        { headers }
      );
      
      graphExecutionTime.add(Date.now() - executeStartTime);
      graphExecutions.add(1);
      
      const executeSuccess = check(executeResponse, {
        'Graph execution initiated': (r) => r.status === 200 || r.status === 402, // 402 = insufficient credits
        'Execution response has ID': (r) => {
          if (r.status !== 200) return true; // Skip check for non-200 responses
          try {
            const execution = JSON.parse(r.body);
            return execution.id !== undefined;
          } catch (e) {
            return false;
          }
        },
      });
      
      if (!executeSuccess) {
        executionErrors.add(1);
        console.log(`❌ Graph execution failed: ${executeResponse.status} - ${executeResponse.body}`);
      } else if (executeResponse.status === 200) {
        try {
          const execution = JSON.parse(executeResponse.body);
          console.log(`✅ Graph execution started: ${execution.id}`);
          
          // 3. Monitor execution status
          monitorExecutionStatus(createdGraph.id, execution.id, headers);
          
        } catch (error) {
          console.error('Error parsing execution response:', error);
          executionErrors.add(1);
        }
      } else if (executeResponse.status === 402) {
        console.log(`⚠️ Insufficient credits for execution`);
      }
      
    } catch (error) {
      console.error('Error handling created graph:', error);
      executionErrors.add(1);
    }
  } else {
    executionErrors.add(1);
    console.log(`❌ Graph creation failed: ${createResponse.status} - ${createResponse.body}`);
  }
}

function complexGraphExecutionJourney(headers) {
  // 20% chance to create and execute complex graph
  if (Math.random() < 0.2) {
    console.log('🔧 Creating complex test graph...');
    const complexGraphData = generateComplexTestGraph();
    
    const createStartTime = Date.now();
    const createResponse = http.post(
      `${config.API_BASE_URL}/api/graphs`,
      JSON.stringify(complexGraphData),
      { headers }
    );
    
    graphCreationTime.add(Date.now() - createStartTime);
    graphCreations.add(1);
    
    const createSuccess = check(createResponse, {
      'Complex graph created successfully': (r) => r.status === 200,
    });
    
    // Debug complex graph creation failures
    if (createResponse.status !== 200) {
      console.log(`❌ Complex graph creation failed: ${createResponse.status} - ${createResponse.body}`);
    }
    
    if (createSuccess && createResponse.status === 200) {
      try {
        const createdGraph = JSON.parse(createResponse.body);
        console.log(`✅ Complex graph created: ${createdGraph.id}`);
        
        // Execute complex graph
        const executionInputs = generateExecutionInputs();
        
        const executeStartTime = Date.now();
        const executeResponse = http.post(
          `${config.API_BASE_URL}/api/graphs/${createdGraph.id}/execute/${createdGraph.version}`,
          JSON.stringify({
            inputs: executionInputs,
            credentials_inputs: {}
          }),
          { headers }
        );
        
        graphExecutionTime.add(Date.now() - executeStartTime);
        graphExecutions.add(1);
        
        check(executeResponse, {
          'Complex graph execution initiated': (r) => r.status === 200 || r.status === 402,
        });
        
        if (executeResponse.status === 200) {
          const execution = JSON.parse(executeResponse.body);
          console.log(`✅ Complex graph execution started: ${execution.id}`);
        }
        
      } catch (error) {
        console.error('Error with complex graph:', error);
        executionErrors.add(1);
      }
    }
  }
}

function graphManagementJourney(headers) {
  // 1. List existing graphs
  const listResponse = http.get(
    `${config.API_BASE_URL}/api/graphs`,
    { headers }
  );
  
  check(listResponse, {
    'Graphs list loaded': (r) => r.status === 200,
  });
  
  // 2. Work with existing graphs (30% chance)
  if (Math.random() < 0.3 && listResponse.status === 200) {
    try {
      const existingGraphs = JSON.parse(listResponse.body);
      
      if (existingGraphs.length > 0) {
        // Pick a random existing graph
        const randomGraph = existingGraphs[Math.floor(Math.random() * existingGraphs.length)];
        
        // Get graph details
        const getResponse = http.get(
          `${config.API_BASE_URL}/api/graphs/${randomGraph.id}`,
          { headers }
        );
        
        check(getResponse, {
          'Existing graph details loaded': (r) => r.status === 200,
        });
        
        // Execute existing graph (50% chance)
        if (Math.random() < 0.5) {
          console.log(`⚡ Executing existing graph: ${randomGraph.id}`);
          
          const executionInputs = generateExecutionInputs();
          
          const executeStartTime = Date.now();
          const executeResponse = http.post(
            `${config.API_BASE_URL}/api/graphs/${randomGraph.id}/execute/${randomGraph.version}`,
            JSON.stringify({
              inputs: executionInputs,
              credentials_inputs: {}
            }),
            { headers }
          );
          
          graphExecutionTime.add(Date.now() - executeStartTime);
          graphExecutions.add(1);
          
          check(executeResponse, {
            'Existing graph execution initiated': (r) => r.status === 200 || r.status === 402,
          });
          
          if (executeResponse.status === 200) {
            const execution = JSON.parse(executeResponse.body);
            console.log(`✅ Existing graph execution: ${execution.id}`);
          }
        }
      }
    } catch (error) {
      console.error('Error working with existing graphs:', error);
    }
  }
}

function monitorExecutionStatus(graphId, executionId, headers) {
  // Wait a bit before checking status
  sleep(1);
  
  const statusResponse = http.get(
    `${config.API_BASE_URL}/api/graphs/${graphId}/executions/${executionId}`,
    { headers }
  );
  
  check(statusResponse, {
    'Execution status retrieved': (r) => r.status === 200,
  });
  
  if (statusResponse.status === 200) {
    try {
      const execution = JSON.parse(statusResponse.body);
      console.log(`📊 Execution status: ${execution.status || 'unknown'}`);
    } catch (error) {
      console.error('Error parsing execution status:', error);
    }
  }
}

export function teardown(data) {
  console.log('🧹 Cleaning up graph execution load test...');
  console.log(`Total graph creations: ${graphCreations.value || 0}`);
  console.log(`Total graph executions: ${graphExecutions.value || 0}`);
  
  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}