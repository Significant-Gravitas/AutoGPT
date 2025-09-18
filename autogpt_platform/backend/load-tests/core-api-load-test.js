// Simple API diagnostic test
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';

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
    console.log(`✅ Auth successful: ${userAuth.access_token.substring(0, 50)}...`);
    
    const headers = getAuthHeaders(userAuth.access_token);
    console.log(`📋 Headers: ${JSON.stringify(headers)}`);
    
    // Step 2: Test profile endpoint with full response logging
    console.log(`🔍 Testing profile endpoint: ${config.API_BASE_URL}/api/auth/user`);
    const profileResponse = http.post(
      `${config.API_BASE_URL}/api/auth/user`,
      '{}',
      { headers }
    );
    
    console.log(`📊 Profile Response:
      Status: ${profileResponse.status}
      Headers: ${JSON.stringify(profileResponse.headers, null, 2)}
      Body: ${profileResponse.body}`);
    
    if (profileResponse.status !== 200) {
      console.log(`❌ Profile API failed with status ${profileResponse.status}`);
      console.log(`❌ Error body: ${profileResponse.body}`);
    }
    
    // Step 3: Test credits endpoint  
    console.log(`💰 Testing credits endpoint: ${config.API_BASE_URL}/api/credits`);
    const creditsResponse = http.get(
      `${config.API_BASE_URL}/api/credits`,
      { headers }
    );
    
    console.log(`📊 Credits Response:
      Status: ${creditsResponse.status}
      Body: ${creditsResponse.body}`);
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
    console.error(`💥 Stack: ${error.stack}`);
  }
}