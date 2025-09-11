import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users over 30s
    { duration: '1m', target: 10 },   // Stay at 10 users for 1 minute
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users for 1 minute
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    errors: ['rate<0.1'],              // Error rate must be below 10%
  },
};

// Base URL for the API
const BASE_URL = __ENV.API_URL || 'https://autogpt-env-staging-significant-gravitas.vercel.app/api/proxy/api/store';

// Test data
const testUserId = 'test-user-' + Math.random().toString(36).substring(7);
const authToken = __ENV.AUTH_TOKEN || ''; // Pass via: k6 run -e AUTH_TOKEN=xxx script.js

export function setup() {
  // Setup code - runs once before the test
  console.log(`Testing Store API at: ${BASE_URL}`);
  
  // Check if API is accessible by getting store agents
  const healthCheck = http.get(`${BASE_URL}/agents?page=1&page_size=1`);
  check(healthCheck, {
    'Store API is accessible': (r) => r.status === 200,
  });
  
  return { baseUrl: BASE_URL, authToken: authToken };
}

export default function (data) {
  // Main test function - this runs for each virtual user
  
  const headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'k6-load-test/1.0',
    'Accept': 'application/json',
    // Add headers to bypass protection if needed
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://autogpt-env-staging-significant-gravitas.vercel.app',
    'Referer': 'https://autogpt-env-staging-significant-gravitas.vercel.app',
  };
  
  // Add auth header if token is provided
  if (data.authToken) {
    headers['Authorization'] = `Bearer ${data.authToken}`;
  }

  // Test 1: Get store agents list
  let response = http.get(`${data.baseUrl}/agents?page=1&page_size=10`, { headers });
  check(response, {
    'store agents status is 200': (r) => r.status === 200,
    'store agents returns data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.agents !== undefined && Array.isArray(body.agents);
      } catch (e) {
        return false;
      }
    },
    'store response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  errorRate.add(response.status !== 200);
  
  sleep(1);

  // Test 2: Get store agents with search
  response = http.get(`${data.baseUrl}/agents?search_query=ai&page=1&page_size=5`, { headers });
  check(response, {
    'search agents status is 200': (r) => r.status === 200,
    'search returns results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.agents !== undefined;
      } catch (e) {
        return false;
      }
    },
    'search response time < 1500ms': (r) => r.timings.duration < 1500,
  });
  errorRate.add(response.status !== 200);
  
  sleep(1);

  // Test 3: Get featured agents
  response = http.get(`${data.baseUrl}/agents?featured=true&page=1&page_size=5`, { headers });
  check(response, {
    'featured agents status is 200': (r) => r.status === 200,
    'featured agents response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  sleep(1);

  // Test 4: Get agents by category
  response = http.get(`${data.baseUrl}/agents?category=productivity&page=1&page_size=5`, { headers });
  check(response, {
    'category filter status is 200': (r) => r.status === 200,
    'category response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  sleep(1);

  // Test 5: Get agents sorted by runs
  response = http.get(`${data.baseUrl}/agents?sorted_by=runs&page=1&page_size=10`, { headers });
  check(response, {
    'sorted agents status is 200': (r) => r.status === 200,
    'sorted agents returns data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.agents !== undefined;
      } catch (e) {
        return false;
      }
    },
    'sorted response time < 1200ms': (r) => r.timings.duration < 1200,
  });
  errorRate.add(response.status !== 200);
  
  sleep(2);

  // Test 6: Test graph operations (if authenticated)
  if (data.authToken) {
    // Create a test graph
    const graphPayload = JSON.stringify({
      name: `Test Graph ${Date.now()}`,
      description: 'K6 load test graph',
      nodes: [],
      edges: [],
    });
    
    response = http.post(`${data.baseUrl}/graphs`, graphPayload, { headers });
    check(response, {
      'create graph successful': (r) => r.status === 200 || r.status === 201,
      'graph creation response time < 1000ms': (r) => r.timings.duration < 1000,
    });
    
    if (response.status === 200 || response.status === 201) {
      const graphId = JSON.parse(response.body).id;
      
      // Get the created graph
      response = http.get(`${data.baseUrl}/graphs/${graphId}`, { headers });
      check(response, {
        'get graph successful': (r) => r.status === 200,
        'get graph response time < 300ms': (r) => r.timings.duration < 300,
      });
      
      // Delete the test graph
      response = http.del(`${data.baseUrl}/graphs/${graphId}`, null, { headers });
      check(response, {
        'delete graph successful': (r) => r.status === 200 || r.status === 204,
      });
    }
  }
  
  sleep(1);
}

export function teardown(data) {
  // Teardown code - runs once after the test
  console.log('Test completed');
}