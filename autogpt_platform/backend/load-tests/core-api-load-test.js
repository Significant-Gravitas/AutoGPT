// Simple API diagnostic test
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { getAuthenticatedUser, getAuthHeaders } from './utils/auth.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [
    { duration: __ENV.RAMP_UP || '30s', target: parseInt(__ENV.VUS) || 1 },
    { duration: __ENV.DURATION || '10s', target: parseInt(__ENV.VUS) || 1 },
    { duration: __ENV.RAMP_DOWN || '30s', target: 0 },
  ],
  thresholds: {
    checks: ['rate>0.85'],
    http_req_duration: ['p(95)<10000'],
    http_req_failed: ['rate<0.15'],
  },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: 'AutoGPT Platform - Core API Validation Test',
  },
};

export default function () {
  try {
    // Step 1: Get authenticated user (cached per VU)
    const userAuth = getAuthenticatedUser();
    const headers = getAuthHeaders(userAuth.access_token);
    console.log(`📋 Headers: ${JSON.stringify(headers)}`);
    
    // Step 2: Test user profile endpoint with full response logging
    console.log(`🔍 Testing user profile endpoint: ${config.API_BASE_URL}/api/auth/user`);
    const profileResponse = http.post(
      `${config.API_BASE_URL}/api/auth/user`,
      '{}',
      { headers }
    );
    
    console.log(`📊 Profile Response:
      Status: ${profileResponse.status}
      Headers: ${JSON.stringify(profileResponse.headers, null, 2)}
      Body: ${profileResponse.body}`);
    
    const profileCheck = check(profileResponse, {
      'Profile API: Status is 200': (r) => r.status === 200,
      'Profile API: Response has user data': (r) => {
        try {
          const data = JSON.parse(r.body);
          return data && data.id; // User should have an ID
        } catch (e) {
          return false;
        }
      },
    });
    
    if (profileResponse.status !== 200) {
      console.log(`❌ Profile API failed with status ${profileResponse.status}`);
      console.log(`❌ Error body: ${profileResponse.body}`);
    } else {
      console.log(`✅ Profile API successful!`);
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
    
    const creditsCheck = check(creditsResponse, {
      'Credits API: Status is 200': (r) => r.status === 200,
      'Credits API: Response has credits': (r) => {
        try {
          const data = JSON.parse(r.body);
          return data && typeof data.credits === 'number';
        } catch (e) {
          return false;
        }
      },
    });
    
    if (creditsResponse.status !== 200) {
      console.log(`❌ Credits API failed with status ${creditsResponse.status}`);
      console.log(`❌ Error body: ${creditsResponse.body}`);
    } else {
      console.log(`✅ Credits API successful!`);
    }
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
    console.error(`💥 Stack: ${error.stack}`);
  }
}