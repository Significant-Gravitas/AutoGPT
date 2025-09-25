// Test individual API endpoints to isolate performance bottlenecks
import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { getAuthenticatedUser, getAuthHeaders } from './utils/auth.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [
    { duration: '10s', target: parseInt(__ENV.VUS) || 3 },
    { duration: '20s', target: parseInt(__ENV.VUS) || 3 },
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    checks: ['rate>0.70'],
    http_req_duration: ['p(95)<5000'],
    http_req_failed: ['rate<0.3'],
  },
};

export default function () {
  const endpoint = __ENV.ENDPOINT || 'credits'; // credits, graphs, blocks, executions
  const concurrentRequests = parseInt(__ENV.CONCURRENT_REQUESTS) || 1;
  
  try {
    const userAuth = getAuthenticatedUser();
    
    if (!userAuth || !userAuth.access_token) {
      console.log(`⚠️ VU ${__VU} has no valid authentication - skipping test`);
      return;
    }
    
    const headers = getAuthHeaders(userAuth.access_token);
    
    console.log(`🚀 VU ${__VU} testing /api/${endpoint} with ${concurrentRequests} concurrent requests`);
    
    if (concurrentRequests === 1) {
      // Single request mode (original behavior)
      const response = http.get(`${config.API_BASE_URL}/api/${endpoint}`, { headers });
      
      const success = check(response, {
        [`${endpoint} API: Status is 200`]: (r) => r.status === 200,
        [`${endpoint} API: Response time < 3s`]: (r) => r.timings.duration < 3000,
      });
      
      if (success) {
        console.log(`✅ VU ${__VU} /api/${endpoint} successful: ${response.timings.duration}ms`);
      } else {
        console.log(`❌ VU ${__VU} /api/${endpoint} failed: ${response.status}, ${response.timings.duration}ms`);
      }
    } else {
      // Concurrent requests mode using http.batch()
      const requests = [];
      for (let i = 0; i < concurrentRequests; i++) {
        requests.push({
          method: 'GET',
          url: `${config.API_BASE_URL}/api/${endpoint}`,
          params: { headers }
        });
      }
      
      const responses = http.batch(requests);
      
      let successCount = 0;
      let totalTime = 0;
      
      for (let i = 0; i < responses.length; i++) {
        const response = responses[i];
        const success = check(response, {
          [`${endpoint} API Request ${i+1}: Status is 200`]: (r) => r.status === 200,
          [`${endpoint} API Request ${i+1}: Response time < 5s`]: (r) => r.timings.duration < 5000,
        });
        
        if (success) {
          successCount++;
        }
        totalTime += response.timings.duration;
      }
      
      const avgTime = totalTime / responses.length;
      console.log(`✅ VU ${__VU} /api/${endpoint}: ${successCount}/${concurrentRequests} successful, avg: ${avgTime.toFixed(0)}ms`);
    }
    
  } catch (error) {
    console.error(`💥 VU ${__VU} error: ${error.message}`);
  }
}