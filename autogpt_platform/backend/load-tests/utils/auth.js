import http from 'k6/http';
import { check, fail } from 'k6';
import { getEnvironmentConfig, AUTH_CONFIG } from '../configs/environment.js';

const config = getEnvironmentConfig();

/**
 * Authenticate user and return JWT token
 * Uses Supabase auth endpoints to get access token
 */
export function authenticateUser(userCredentials) {
  // Supabase auth login endpoint
  const authUrl = `${config.SUPABASE_URL}/auth/v1/token?grant_type=password`;
  
  const loginPayload = {
    email: userCredentials.email,
    password: userCredentials.password,
  };
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'apikey': config.SUPABASE_ANON_KEY,
    },
    timeout: '30s', // Add timeout to prevent hanging
  };
  
  // Retry logic for network reliability
  let response;
  let authSuccess = false;
  let lastError = '';
  
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      response = http.post(authUrl, JSON.stringify(loginPayload), params);
      
      authSuccess = check(response, {
        'Authentication successful': (r) => r.status === 200,
        'Auth response has access token': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.access_token !== undefined;
          } catch (e) {
            return false;
          }
        },
      });
      
      if (authSuccess) {
        break; // Success, exit retry loop
      }
      
      lastError = `${response.status} ${response.body}`;
      console.log(`Authentication attempt ${attempt} failed for ${userCredentials.email}: ${lastError}`);
      
    } catch (error) {
      lastError = error.message;
      console.log(`Authentication attempt ${attempt} error for ${userCredentials.email}: ${lastError}`);
    }
    
    if (attempt < 3) {
      console.log(`Retrying authentication in 1 second...`);
      // Small delay before retry
      http.get('https://httpbin.org/delay/1');
    }
  }
  
  if (!authSuccess) {
    console.log(`Authentication failed for ${userCredentials.email} after 3 attempts: ${lastError}`);
    fail('Authentication failed');
  }
  
  const authData = JSON.parse(response.body);
  return {
    access_token: authData.access_token,
    refresh_token: authData.refresh_token,
    user: authData.user,
  };
}

/**
 * Get authenticated headers for API requests
 */
export function getAuthHeaders(accessToken) {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`,
  };
}

/**
 * Get random test user credentials
 */
export function getRandomTestUser() {
  const users = AUTH_CONFIG.TEST_USERS;
  return users[Math.floor(Math.random() * users.length)];
}