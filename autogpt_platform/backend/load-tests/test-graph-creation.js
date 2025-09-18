// Test graph creation with correct block IDs
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';
import { generateTestGraph } from './utils/test-data.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [{ duration: '10s', target: 1 }],
};

export default function () {
  const testUser = getRandomTestUser();
  
  try {
    // Step 1: Authenticate
    console.log(`🔐 Authenticating ${testUser.email}...`);
    const userAuth = authenticateUser(testUser);
    const headers = getAuthHeaders(userAuth.access_token);
    
    // Step 2: Create a test graph
    console.log(`📊 Creating test graph...`);
    const graphData = generateTestGraph();
    console.log(`Graph data: ${JSON.stringify(graphData, null, 2)}`);
    
    const createResponse = http.post(
      `${config.API_BASE_URL}/api/graphs`,
      JSON.stringify(graphData),
      { headers }
    );
    
    console.log(`📊 Graph Creation Response:`);
    console.log(`   Status: ${createResponse.status}`);
    console.log(`   Body: ${createResponse.body}`);
    
    const success = check(createResponse, {
      'Graph created successfully': (r) => r.status === 200,
    });
    
    if (!success) {
      console.log(`❌ Graph creation failed`);
      try {
        const errorData = JSON.parse(createResponse.body);
        console.log(`   Error: ${JSON.stringify(errorData, null, 2)}`);
      } catch (e) {
        console.log(`   Raw error: ${createResponse.body}`);
      }
    } else {
      console.log(`✅ Graph created successfully!`);
    }
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
  }
}