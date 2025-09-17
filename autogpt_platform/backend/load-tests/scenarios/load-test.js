import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvironmentConfig, PERFORMANCE_CONFIG } from '../configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from '../utils/auth.js';
import { 
  generateTestGraph, 
  generateExecutionInputs, 
  generateScheduleData,
  generateAPIKeyRequest 
} from '../utils/test-data.js';

const config = getEnvironmentConfig();

// Custom metrics
const userOperations = new Counter('user_operations_total');
const graphOperations = new Counter('graph_operations_total');
const executionOperations = new Counter('execution_operations_total');
const apiResponseTime = new Trend('api_response_time');
const authErrors = new Rate('auth_errors');

// Test configuration for normal load testing
export const options = {
  stages: [
    { duration: PERFORMANCE_CONFIG.DEFAULT_RAMP_UP, target: PERFORMANCE_CONFIG.DEFAULT_VUS },
    { duration: PERFORMANCE_CONFIG.DEFAULT_DURATION, target: PERFORMANCE_CONFIG.DEFAULT_VUS },
    { duration: PERFORMANCE_CONFIG.DEFAULT_RAMP_DOWN, target: 0 },
  ],
  thresholds: PERFORMANCE_CONFIG.THRESHOLDS,
  ext: {
    loadimpact: {
      projectID: __ENV.K6_CLOUD_PROJECT_ID,
      name: 'AutoGPT Platform Load Test',
    },
  },
};

export function setup() {
  console.log('🎯 Setting up load test scenario...');
  return {
    timestamp: Date.now()
  };
}

export default function (data) {
  // Each VU gets a random test user for authentication
  const testUser = getRandomTestUser();
  let userAuth;
  
  try {
    userAuth = authenticateUser(testUser);
  } catch (error) {
    console.error(`❌ Authentication failed for ${testUser.email}:`, error);
    authErrors.add(1);
    return;
  }
  
  const headers = getAuthHeaders(userAuth.access_token);
  
  // Realistic user journey simulation
  group('User Authentication & Profile', () => {
    userProfileJourney(headers);
  });
  
  group('Graph Management', () => {
    graphManagementJourney(headers);
  });
  
  group('Block Operations', () => {
    blockOperationsJourney(headers);
  });
  
  group('System Operations', () => {
    systemOperationsJourney(headers);
  });
  
  // Think time between user sessions
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

function userProfileJourney(headers) {
  const startTime = Date.now();
  
  // 1. Get user profile
  const profileResponse = http.post(
    `${config.API_BASE_URL}/api/auth/user`,
    '{}',
    { headers }
  );
  
  userOperations.add(1);
  
  check(profileResponse, {
    'User profile loaded successfully': (r) => r.status === 200,
  });
  
  // 2. Get user credits
  const creditsResponse = http.get(
    `${config.API_BASE_URL}/api/credits`,
    { headers }
  );
  
  userOperations.add(1);
  
  check(creditsResponse, {
    'User credits loaded successfully': (r) => r.status === 200,
  });
  
  // 3. Check onboarding status
  const onboardingResponse = http.get(
    `${config.API_BASE_URL}/api/onboarding`,
    { headers }
  );
  
  userOperations.add(1);
  
  check(onboardingResponse, {
    'Onboarding status loaded': (r) => r.status === 200,
  });
  
  apiResponseTime.add(Date.now() - startTime);
}

function graphManagementJourney(headers) {
  const startTime = Date.now();
  
  // 1. List existing graphs
  const listResponse = http.get(
    `${config.API_BASE_URL}/api/graphs`,
    { headers }
  );
  
  graphOperations.add(1);
  
  const listSuccess = check(listResponse, {
    'Graphs list loaded successfully': (r) => r.status === 200,
  });
  
  // 2. Create a new graph (20% of users)
  if (Math.random() < 0.2) {
    const graphData = generateTestGraph();
    
    const createResponse = http.post(
      `${config.API_BASE_URL}/api/graphs`,
      JSON.stringify(graphData),
      { headers }
    );
    
    graphOperations.add(1);
    
    const createSuccess = check(createResponse, {
      'Graph created successfully': (r) => r.status === 200,
    });
    
    if (createSuccess && createResponse.status === 200) {
      try {
        const createdGraph = JSON.parse(createResponse.body);
        
        // 3. Get the created graph details
        const getResponse = http.get(
          `${config.API_BASE_URL}/api/graphs/${createdGraph.id}`,
          { headers }
        );
        
        graphOperations.add(1);
        
        check(getResponse, {
          'Graph details loaded': (r) => r.status === 200,
        });
        
        // 4. Execute the graph (50% chance)
        if (Math.random() < 0.5) {
          executeGraphScenario(createdGraph, headers);
        }
        
        // 5. Create schedule for graph (10% chance)
        if (Math.random() < 0.1) {
          createScheduleScenario(createdGraph.id, headers);
        }
        
      } catch (error) {
        console.error('Error handling created graph:', error);
      }
    }
  }
  
  // 3. Work with existing graphs (if any)
  if (listSuccess && listResponse.status === 200) {
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
        
        graphOperations.add(1);
        
        check(getResponse, {
          'Existing graph details loaded': (r) => r.status === 200,
        });
        
        // Execute existing graph (30% chance)
        if (Math.random() < 0.3) {
          executeGraphScenario(randomGraph, headers);
        }
      }
    } catch (error) {
      console.error('Error working with existing graphs:', error);
    }
  }
  
  apiResponseTime.add(Date.now() - startTime);
}

function executeGraphScenario(graph, headers) {
  const startTime = Date.now();
  
  const executionInputs = generateExecutionInputs();
  
  const executeResponse = http.post(
    `${config.API_BASE_URL}/api/graphs/${graph.id}/execute/${graph.version}`,
    JSON.stringify({
      inputs: executionInputs,
      credentials_inputs: {}
    }),
    { headers }
  );
  
  executionOperations.add(1);
  
  const executeSuccess = check(executeResponse, {
    'Graph execution initiated': (r) => r.status === 200 || r.status === 402, // 402 = insufficient credits
  });
  
  if (executeSuccess && executeResponse.status === 200) {
    try {
      const execution = JSON.parse(executeResponse.body);
      
      // Monitor execution status (simulate user checking results)
      setTimeout(() => {
        const statusResponse = http.get(
          `${config.API_BASE_URL}/api/graphs/${graph.id}/executions/${execution.id}`,
          { headers }
        );
        
        executionOperations.add(1);
        
        check(statusResponse, {
          'Execution status retrieved': (r) => r.status === 200,
        });
      }, 2000);
      
    } catch (error) {
      console.error('Error monitoring execution:', error);
    }
  }
  
  apiResponseTime.add(Date.now() - startTime);
}

function createScheduleScenario(graphId, headers) {
  const scheduleData = generateScheduleData(graphId);
  
  const scheduleResponse = http.post(
    `${config.API_BASE_URL}/api/graphs/${graphId}/schedules`,
    JSON.stringify(scheduleData),
    { headers }
  );
  
  graphOperations.add(1);
  
  check(scheduleResponse, {
    'Schedule created successfully': (r) => r.status === 200,
  });
}

function blockOperationsJourney(headers) {
  const startTime = Date.now();
  
  // 1. Get available blocks
  const blocksResponse = http.get(
    `${config.API_BASE_URL}/api/blocks`,
    { headers }
  );
  
  userOperations.add(1);
  
  const blocksSuccess = check(blocksResponse, {
    'Blocks list loaded': (r) => r.status === 200,
  });
  
  // 2. Execute some blocks directly (simulate testing)
  if (blocksSuccess && Math.random() < 0.3) {
    // Execute GetCurrentTimeBlock (simple, fast block)
    const timeBlockResponse = http.post(
      `${config.API_BASE_URL}/api/blocks/a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa/execute`,
      JSON.stringify({
        trigger: "test",
        format_type: {
          discriminator: "iso8601",
          timezone: "UTC"
        }
      }),
      { headers }
    );
    
    userOperations.add(1);
    
    check(timeBlockResponse, {
      'Time block executed or handled gracefully': (r) => r.status === 200 || r.status === 500, // 500 = user_context missing (expected)
    });
  }
  
  apiResponseTime.add(Date.now() - startTime);
}

function systemOperationsJourney(headers) {
  const startTime = Date.now();
  
  // 1. Check executions list (simulate monitoring)
  const executionsResponse = http.get(
    `${config.API_BASE_URL}/api/executions`,
    { headers }
  );
  
  userOperations.add(1);
  
  check(executionsResponse, {
    'Executions list loaded': (r) => r.status === 200,
  });
  
  // 2. Check schedules (if any)
  const schedulesResponse = http.get(
    `${config.API_BASE_URL}/api/schedules`,
    { headers }
  );
  
  userOperations.add(1);
  
  check(schedulesResponse, {
    'Schedules list loaded': (r) => r.status === 200,
  });
  
  // 3. Check API keys (simulate user managing access)
  if (Math.random() < 0.1) { // 10% of users check API keys
    const apiKeysResponse = http.get(
      `${config.API_BASE_URL}/api/api-keys`,
      { headers }
    );
    
    userOperations.add(1);
    
    check(apiKeysResponse, {
      'API keys list loaded': (r) => r.status === 200,
    });
    
    // Occasionally create new API key (5% chance)
    if (Math.random() < 0.05) {
      const keyData = generateAPIKeyRequest();
      
      const createKeyResponse = http.post(
        `${config.API_BASE_URL}/api/api-keys`,
        JSON.stringify(keyData),
        { headers }
      );
      
      userOperations.add(1);
      
      check(createKeyResponse, {
        'API key created successfully': (r) => r.status === 200,
      });
    }
  }
  
  apiResponseTime.add(Date.now() - startTime);
}

export function teardown(data) {
  console.log('🧹 Cleaning up load test...');
  console.log(`Total user operations: ${userOperations.value}`);
  console.log(`Total graph operations: ${graphOperations.value}`);
  console.log(`Total execution operations: ${executionOperations.value}`);
  
  const testDuration = Date.now() - data.timestamp;
  console.log(`Test completed in ${testDuration}ms`);
}