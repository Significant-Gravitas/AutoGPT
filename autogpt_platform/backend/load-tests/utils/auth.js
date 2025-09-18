import http from 'k6/http';
import { check, fail } from 'k6';
import { getEnvironmentConfig, AUTH_CONFIG } from '../configs/environment.js';

const config = getEnvironmentConfig();

// VU-specific token cache to avoid re-authentication
const vuTokenCache = new Map();

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
    timeout: '30s',
  };
  
  // Single authentication attempt - no retries to avoid amplifying rate limits
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
    console.log(`❌ Auth failed for ${userCredentials.email}: ${response.status} - ${response.body.substring(0, 200)}`);
    return null; // Return null instead of failing the test
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

/**
 * Smart authentication that caches tokens per VU to avoid rate limiting
 * Authenticates once per VU and reuses the token across iterations
 */
export function getAuthenticatedUser() {
  const vuId = __VU; // k6 VU identifier
  
  // Check if we already have a valid token for this VU
  if (vuTokenCache.has(vuId)) {
    const cachedAuth = vuTokenCache.get(vuId);
    console.log(`🔄 Using cached token for VU ${vuId} (user: ${cachedAuth.user.email})`);
    return cachedAuth;
  }
  
  // Try to authenticate with available test users
  const users = AUTH_CONFIG.TEST_USERS;
  let authResult = null;
  
  for (let i = 0; i < users.length; i++) {
    const testUser = users[i];
    console.log(`🔐 VU ${vuId} attempting authentication with ${testUser.email}...`);
    
    authResult = authenticateUser(testUser);
    
    if (authResult) {
      // Cache the successful authentication for this VU
      vuTokenCache.set(vuId, authResult);
      console.log(`✅ VU ${vuId} authenticated successfully with ${testUser.email}`);
      return authResult;
    }
    
    console.log(`❌ VU ${vuId} authentication failed with ${testUser.email}, trying next user...`);
  }
  
  // If all users failed, throw error
  throw new Error(`VU ${vuId} failed to authenticate with any test user`);
}

/**
 * Clear authentication cache (useful for testing or cleanup)
 */
export function clearAuthCache() {
  vuTokenCache.clear();
  console.log('🧹 Authentication cache cleared');
}