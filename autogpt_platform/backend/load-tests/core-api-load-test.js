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
  // Get load multiplier - how many concurrent requests each VU should make
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;
  
  try {
    // Step 1: Get authenticated user (cached per VU)
    const userAuth = getAuthenticatedUser();
    const headers = getAuthHeaders(userAuth.access_token);
    
    console.log(`🚀 VU ${__VU} making ${requestsPerVU} concurrent API requests...`);
    
    // Create array of API requests to run concurrently
    const requests = [];
    
    for (let i = 0; i < requestsPerVU; i++) {
      // Add profile API request
      requests.push({
        method: 'POST',
        url: `${config.API_BASE_URL}/api/auth/user`,
        body: '{}',
        params: { headers }
      });
      
      // Add credits API request  
      requests.push({
        method: 'GET',
        url: `${config.API_BASE_URL}/api/credits`,
        params: { headers }
      });
    }
    
    // Execute all requests concurrently
    const responses = http.batch(requests);
    
    // Validate results
    let profileSuccesses = 0;
    let creditsSuccesses = 0;
    
    for (let i = 0; i < responses.length; i++) {
      const response = responses[i];
      
      if (i % 2 === 0) {
        // Profile API request
        const profileCheck = check(response, {
          'Profile API: Status is 200': (r) => r.status === 200,
          'Profile API: Response has user data': (r) => {
            try {
              const data = JSON.parse(r.body);
              return data && data.id;
            } catch (e) {
              return false;
            }
          },
        });
        if (profileCheck) profileSuccesses++;
      } else {
        // Credits API request
        const creditsCheck = check(response, {
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
        if (creditsCheck) creditsSuccesses++;
      }
    }
    
    console.log(`✅ VU ${__VU} completed: ${profileSuccesses}/${requestsPerVU} profile, ${creditsSuccesses}/${requestsPerVU} credits requests successful`);
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
    console.error(`💥 Stack: ${error.stack}`);
  }
}