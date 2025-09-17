import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvironmentConfig, PERFORMANCE_CONFIG } from '../configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from '../utils/auth.js';
import { 
  generateTestGraph, 
  generateExecutionInputs, 
  generateBlockExecutionData,
  generateAPIKeyRequest,
  generateTopUpRequest 
} from '../utils/test-data.js';

const config = getEnvironmentConfig();

// Custom metrics
const authErrors = new Rate('auth_errors');
const graphCreationTime = new Trend('graph_creation_time');
const executionTime = new Trend('execution_time');
const blockExecutionTime = new Trend('block_execution_time');
const apiRequests = new Counter('api_requests_total');

// Test configuration
export const options = {
  stages: [
    { duration: PERFORMANCE_CONFIG.STRESS_RAMP_UP, target: PERFORMANCE_CONFIG.STRESS_VUS },
    { duration: PERFORMANCE_CONFIG.STRESS_DURATION, target: PERFORMANCE_CONFIG.STRESS_VUS },
    { duration: PERFORMANCE_CONFIG.STRESS_RAMP_DOWN, target: 0 },
  ],
  thresholds: {
    ...PERFORMANCE_CONFIG.THRESHOLDS,
    auth_errors: ['rate<0.1'],
    graph_creation_time: ['p(95)<5000'],
    execution_time: ['p(95)<10000'],
    block_execution_time: ['p(95)<3000'],
  },
  ext: {
    loadimpact: {
      projectID: __ENV.K6_CLOUD_PROJECT_ID,
      name: 'AutoGPT Platform API Stress Test',
    },
  },
};

// Global test state
let userAuth = null;

export function setup() {
  console.log('🚀 Setting up API stress test...');
  
  // Authenticate a test user for the test
  const testUser = getRandomTestUser();
  console.log(`Authenticating user: ${testUser.email}`);
  
  try {
    userAuth = authenticateUser(testUser);
    console.log('✅ Authentication successful');
    return { userAuth };
  } catch (error) {
    console.error('❌ Setup authentication failed:', error);
    throw error;
  }
}

export default function (data) {
  const { userAuth } = data;
  
  if (!userAuth || !userAuth.access_token) {
    console.error('❌ No valid authentication token available');
    authErrors.add(1);
    return;
  }
  
  const headers = getAuthHeaders(userAuth.access_token);
  
  // Test scenario: API endpoint stress testing
  apiStressTestScenario(headers);
  
  sleep(1); // Brief pause between iterations
}

function apiStressTestScenario(headers) {
  // 1. Test user profile endpoint
  testUserProfile(headers);
  
  // 2. Test blocks listing
  testBlocksListing(headers);
  
  // 3. Test graph operations
  testGraphOperations(headers);
  
  // 4. Test block execution
  testBlockExecution(headers);
  
  // 5. Test credits system
  testCreditsSystem(headers);
  
  // 6. Test file operations
  testFileOperations(headers);
}

function testUserProfile(headers) {
  const startTime = Date.now();
  
  // Get user profile
  const profileResponse = http.post(
    `${config.API_BASE_URL}/api/v1/auth/user`,
    '{}',
    { headers }
  );
  
  apiRequests.add(1);
  
  check(profileResponse, {
    'Get user profile - status 200': (r) => r.status === 200,
    'Get user profile - has user data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  // Get user credits
  const creditsResponse = http.get(
    `${config.API_BASE_URL}/api/v1/credits`,
    { headers }
  );
  
  apiRequests.add(1);
  
  check(creditsResponse, {
    'Get credits - status 200': (r) => r.status === 200,
    'Get credits - has credits field': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.credits !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  const endTime = Date.now();
  console.log(`User profile test completed in ${endTime - startTime}ms`);
}

function testBlocksListing(headers) {
  const startTime = Date.now();
  
  const blocksResponse = http.get(
    `${config.API_BASE_URL}/api/v1/blocks`,
    { headers }
  );
  
  apiRequests.add(1);
  
  check(blocksResponse, {
    'Get blocks - status 200': (r) => r.status === 200,
    'Get blocks - returns array': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body) && body.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
  
  blockExecutionTime.add(Date.now() - startTime);
}

function testGraphOperations(headers) {
  const startTime = Date.now();
  
  // 1. List existing graphs
  const listResponse = http.get(
    `${config.API_BASE_URL}/api/v1/graphs`,
    { headers }
  );
  
  apiRequests.add(1);
  
  check(listResponse, {
    'List graphs - status 200': (r) => r.status === 200,
  });
  
  // 2. Create a new graph
  const graphData = generateTestGraph();
  const createResponse = http.post(
    `${config.API_BASE_URL}/api/v1/graphs`,
    JSON.stringify(graphData),
    { headers }
  );
  
  apiRequests.add(1);
  
  const graphCreated = check(createResponse, {
    'Create graph - status 200': (r) => r.status === 200,
    'Create graph - returns graph ID': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  graphCreationTime.add(Date.now() - startTime);
  
  if (graphCreated && createResponse.status === 200) {
    try {
      const createdGraph = JSON.parse(createResponse.body);
      
      // 3. Get the created graph
      const getResponse = http.get(
        `${config.API_BASE_URL}/api/v1/graphs/${createdGraph.id}`,
        { headers }
      );
      
      apiRequests.add(1);
      
      check(getResponse, {
        'Get graph - status 200': (r) => r.status === 200,
      });
      
      // 4. Execute the graph
      const executionInputs = generateExecutionInputs();
      const executeResponse = http.post(
        `${config.API_BASE_URL}/api/v1/graphs/${createdGraph.id}/execute/${createdGraph.version}`,
        JSON.stringify({
          inputs: executionInputs,
          credentials_inputs: {}
        }),
        { headers }
      );
      
      apiRequests.add(1);
      
      check(executeResponse, {
        'Execute graph - status 200 or 402': (r) => r.status === 200 || r.status === 402, // 402 = insufficient credits
      });
      
      executionTime.add(Date.now() - startTime);
      
    } catch (error) {
      console.error('Error in graph operations:', error);
    }
  }
}

function testBlockExecution(headers) {
  const startTime = Date.now();
  
  // Test executing a simple block (GetCurrentTimeBlock)
  const blockData = generateBlockExecutionData('GetCurrentTimeBlock');
  
  const executeResponse = http.post(
    `${config.API_BASE_URL}/api/v1/blocks/GetCurrentTimeBlock/execute`,
    JSON.stringify(blockData),
    { headers }
  );
  
  apiRequests.add(1);
  
  check(executeResponse, {
    'Execute block - status 200': (r) => r.status === 200,
    'Execute block - has output': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Object.keys(body).length > 0;
      } catch (e) {
        return false;
      }
    },
  });
  
  blockExecutionTime.add(Date.now() - startTime);
}

function testCreditsSystem(headers) {
  // Get credit history
  const historyResponse = http.get(
    `${config.API_BASE_URL}/api/v1/credits/transactions?transaction_count_limit=10`,
    { headers }
  );
  
  apiRequests.add(1);
  
  check(historyResponse, {
    'Get credit history - status 200': (r) => r.status === 200,
  });
  
  // Test top-up intent (won't actually charge)
  const topUpData = generateTopUpRequest();
  const topUpResponse = http.post(
    `${config.API_BASE_URL}/api/v1/credits`,
    JSON.stringify(topUpData),
    { headers }
  );
  
  apiRequests.add(1);
  
  check(topUpResponse, {
    'Credit top-up intent - status 200': (r) => r.status === 200,
    'Credit top-up - returns checkout URL': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.checkout_url !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
}

function testFileOperations(headers) {
  // Test file upload with small test content
  const testFileContent = 'Load test file content';
  const uploadData = {
    file: http.file(testFileContent, 'test-file.txt', 'text/plain'),
    provider: 'gcs',
    expiration_hours: 24
  };
  
  const uploadResponse = http.post(
    `${config.API_BASE_URL}/api/v1/files/upload`,
    uploadData,
    { headers }
  );
  
  apiRequests.add(1);
  
  check(uploadResponse, {
    'File upload - status 200': (r) => r.status === 200,
    'File upload - returns file URI': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.file_uri !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
}

export function teardown(data) {
  console.log('🧹 Cleaning up API stress test...');
  console.log(`Total API requests made: ${apiRequests.value}`);
}