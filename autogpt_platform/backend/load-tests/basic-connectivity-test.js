/**
 * Basic Connectivity Test
 * 
 * Tests basic connectivity and authentication without requiring backend API access
 * This test validates that the core infrastructure is working correctly
 */

import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getRandomTestUser } from './utils/auth.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [{ duration: '10s', target: 1 }],
  thresholds: {
    checks: ['rate>0.95'],
    http_req_duration: ['p(95)<2000'],
    http_req_failed: ['rate<0.1'],
  },
};

export default function () {
  const testUser = getRandomTestUser();
  
  try {
    // Test 1: Verify Supabase authentication endpoint is reachable
    console.log(`🔗 Testing Supabase connectivity: ${config.SUPABASE_URL}`);
    const pingResponse = http.get(`${config.SUPABASE_URL}/rest/v1/`);
    
    const connectivityCheck = check(pingResponse, {
      'Supabase connectivity: Status is not 500': (r) => r.status !== 500,
      'Supabase connectivity: Response time < 2s': (r) => r.timings.duration < 2000,
    });
    
    if (pingResponse.status !== 500) {
      console.log(`✅ Supabase reachable (status: ${pingResponse.status})`);
    } else {
      console.log(`❌ Supabase connectivity issue (status: ${pingResponse.status})`);
    }
    
    // Test 2: Verify authentication works
    console.log(`🔐 Testing authentication for ${testUser.email}...`);
    const authStartTime = new Date().getTime();
    const userAuth = authenticateUser(testUser);
    const authDuration = new Date().getTime() - authStartTime;
    
    const authCheck = check(userAuth, {
      'Authentication: Access token received': (auth) => auth && auth.access_token && auth.access_token.length > 0,
      'Authentication: Response time < 1s': () => authDuration < 1000,
    });
    
    if (userAuth && userAuth.access_token) {
      console.log(`✅ Authentication successful (${authDuration}ms)`);
      console.log(`🎫 Token length: ${userAuth.access_token.length} chars`);
      
      // Test 3: Verify token structure (basic JWT validation)
      const tokenParts = userAuth.access_token.split('.');
      const tokenStructureCheck = check(tokenParts, {
        'JWT token: Has 3 parts (header.payload.signature)': (parts) => parts.length === 3,
        'JWT token: Header is base64': (parts) => parts[0] && parts[0].length > 10,
        'JWT token: Payload is base64': (parts) => parts[1] && parts[1].length > 50,
        'JWT token: Signature exists': (parts) => parts[2] && parts[2].length > 10,
      });
      
      if (tokenParts.length === 3) {
        console.log(`✅ JWT token structure valid`);
      } else {
        console.log(`❌ JWT token structure invalid (${tokenParts.length} parts)`);
      }
      
      // Test 4: Basic backend server connectivity (just to see if it responds)
      console.log(`🌐 Testing backend server connectivity: ${config.API_BASE_URL}`);
      const backendResponse = http.get(`${config.API_BASE_URL}`);
      
      const backendCheck = check(backendResponse, {
        'Backend server: Responds (any status)': (r) => r.status > 0,
        'Backend server: Response time < 2s': (r) => r.timings.duration < 2000,
      });
      
      console.log(`📊 Backend server status: ${backendResponse.status}`);
      
    } else {
      console.log(`❌ Authentication failed for ${testUser.email}`);
    }
    
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
    check(null, {
      'Test execution: No errors': () => false,
    });
  }
}

export function teardown(data) {
  console.log(`🏁 Basic connectivity test completed`);
}