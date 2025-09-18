// Diagnostic test to identify 400/500 errors
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';
import { generateTestGraph, generateExecutionInputs } from './utils/test-data.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [{ duration: '20s', target: 1 }],
};

export default function () {
  const testUser = getRandomTestUser();
  
  try {
    // Step 1: Authenticate
    console.log(`🔐 Authenticating ${testUser.email}...`);
    const userAuth = authenticateUser(testUser);
    const headers = getAuthHeaders(userAuth.access_token);
    
    // Step 2: Test each endpoint individually and log errors
    
    // Test profile (should work)
    console.log(`\n=== Testing Profile API ===`);
    testEndpoint('POST', `${config.API_BASE_URL}/api/auth/user`, '{}', headers);
    
    // Test credits (should work)
    console.log(`\n=== Testing Credits API ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/credits`, null, headers);
    
    // Test graphs list (might work)
    console.log(`\n=== Testing Graphs List ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/graphs`, null, headers);
    
    // Test blocks list (might work)
    console.log(`\n=== Testing Blocks List ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/blocks`, null, headers);
    
    // Test block execution (likely failing)
    console.log(`\n=== Testing Block Execution ===`);
    testEndpoint('POST', `${config.API_BASE_URL}/api/blocks/a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa/execute`, 
      JSON.stringify({ 
        trigger: "test",
        format_type: {
          discriminator: "iso8601",
          timezone: "UTC"
        }
      }), headers);
    
    // Test executions list (might work)
    console.log(`\n=== Testing Executions List ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/executions`, null, headers);
    
    // Test schedules list (might work)
    console.log(`\n=== Testing Schedules List ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/schedules`, null, headers);
    
    // Test onboarding (might work)
    console.log(`\n=== Testing Onboarding ===`);
    testEndpoint('GET', `${config.API_BASE_URL}/api/onboarding`, null, headers);
    
    // Test graph creation and execution (comprehensive)
    console.log(`\n=== Testing Graph Creation ===`);
    const graphData = generateTestGraph();
    const createGraphResponse = testEndpoint('POST', `${config.API_BASE_URL}/api/graphs`, 
      JSON.stringify(graphData), headers);
    
    // If graph creation succeeds, test graph execution
    if (createGraphResponse && createGraphResponse.status === 200) {
      try {
        const createdGraph = JSON.parse(createGraphResponse.body);
        console.log(`✅ Graph created successfully: ${createdGraph.id}`);
        
        console.log(`\n=== Testing Graph Execution ===`);
        const executionInputs = generateExecutionInputs();
        testEndpoint('POST', `${config.API_BASE_URL}/api/graphs/${createdGraph.id}/execute/${createdGraph.version}`,
          JSON.stringify({
            inputs: executionInputs,
            credentials_inputs: {}
          }), headers);
          
        console.log(`\n=== Testing Graph Details ===`);
        testEndpoint('GET', `${config.API_BASE_URL}/api/graphs/${createdGraph.id}`, null, headers);
        
      } catch (e) {
        console.log(`❌ Failed to parse created graph response: ${e.message}`);
      }
    }
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
  }
}

function testEndpoint(method, url, body, headers) {
  let response;
  
  if (method === 'GET') {
    response = http.get(url, { headers });
  } else {
    response = http.post(url, body, { headers });
  }
  
  const status = response.status;
  const responseBody = response.body;
  
  console.log(`${method} ${url.replace(config.API_BASE_URL, '')}`);
  console.log(`Status: ${status}`);
  
  if (status >= 400) {
    console.log(`❌ ERROR Response:`);
    console.log(`Headers: ${JSON.stringify(response.headers, null, 2)}`);
    console.log(`Body: ${responseBody}`);
    
    // Try to parse error details
    try {
      const errorData = JSON.parse(responseBody);
      if (errorData.detail) {
        console.log(`Error Detail: ${errorData.detail}`);
      }
      if (errorData.message) {
        console.log(`Error Message: ${errorData.message}`);
      }
      if (errorData.errors) {
        console.log(`Validation Errors: ${JSON.stringify(errorData.errors, null, 2)}`);
      }
    } catch (e) {
      console.log(`Raw error body: ${responseBody}`);
    }
  } else {
    console.log(`✅ SUCCESS (${responseBody.length} bytes)`);
    // For successful responses, show a preview
    if (responseBody.length < 200) {
      console.log(`Response: ${responseBody}`);
    } else {
      console.log(`Response: ${responseBody.substring(0, 100)}...`);
    }
  }
  console.log(`---`);
  
  return response;
}