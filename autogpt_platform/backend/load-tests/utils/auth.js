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
  };
  
  const response = http.post(authUrl, JSON.stringify(loginPayload), params);
  
  const authSuccess = check(response, {
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
  
  if (!authSuccess) {
    console.log(`Authentication failed for ${userCredentials.email}: ${response.status} ${response.body}`);
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