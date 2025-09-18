// Focused test to identify all failing requests
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [
    { duration: '30s', target: 5 }, // Moderate load to surface failures
  ],
};

export default function () {
  const testUser = getRandomTestUser();
  
  try {
    // Step 1: Authenticate
    const userAuth = authenticateUser(testUser);
    const headers = getAuthHeaders(userAuth.access_token);
    
    // Step 2: Test each endpoint and track failures
    const endpoints = [
      { method: 'POST', url: '/api/auth/user', body: '{}', name: 'Profile' },
      { method: 'GET', url: '/api/credits', body: null, name: 'Credits' },
      { method: 'GET', url: '/api/graphs', body: null, name: 'Graphs' },
      { method: 'GET', url: '/api/executions', body: null, name: 'Executions' },
      { method: 'GET', url: '/api/schedules', body: null, name: 'Schedules' },
      { method: 'GET', url: '/api/onboarding', body: null, name: 'Onboarding' },
    ];
    
    for (const endpoint of endpoints) {
      let response;
      const fullUrl = `${config.API_BASE_URL}${endpoint.url}`;
      
      if (endpoint.method === 'GET') {
        response = http.get(fullUrl, { headers });
      } else {
        response = http.post(fullUrl, endpoint.body, { headers });
      }
      
      const success = check(response, {
        [`${endpoint.name} API success`]: (r) => r.status === 200,
      });
      
      if (!success) {
        console.log(`❌ FAILURE: ${endpoint.name} ${endpoint.method} ${endpoint.url}`);
        console.log(`   Status: ${response.status}`);
        console.log(`   Body: ${response.body.substring(0, 200)}...`);
      }
    }
    
  } catch (error) {
    console.error(`💥 Authentication failed for ${testUser.email}: ${error.message}`);
  }
}