import http from 'k6/http';
import { check, fail, sleep } from 'k6';
import { getEnvironmentConfig, AUTH_CONFIG } from '../configs/environment.js';

const config = getEnvironmentConfig();

// VU-specific token cache to avoid re-authentication
const vuTokenCache = new Map();

// Batch authentication coordination for high VU counts
let currentBatch = 0;
let batchAuthInProgress = false;
const BATCH_SIZE = 30; // Respect Supabase rate limit
const authQueue = [];
let authQueueProcessing = false;

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
 * Smart authentication with batch processing for high VU counts
 * Processes authentication in batches of 30 to respect rate limits
 */
export function getAuthenticatedUser() {
  const vuId = __VU; // k6 VU identifier
  
  // Check if we already have a valid token for this VU
  if (vuTokenCache.has(vuId)) {
    const cachedAuth = vuTokenCache.get(vuId);
    console.log(`🔄 Using cached token for VU ${vuId} (user: ${cachedAuth.user.email})`);
    return cachedAuth;
  }
  
  // Use batch authentication for high VU counts
  return batchAuthenticate(vuId);
}

/**
 * Batch authentication processor that handles VUs in groups of 30
 * This respects Supabase's rate limit while allowing higher concurrency
 */
function batchAuthenticate(vuId) {
  const users = AUTH_CONFIG.TEST_USERS;
  
  // Determine which batch this VU belongs to
  const batchNumber = Math.floor((vuId - 1) / BATCH_SIZE);
  const positionInBatch = ((vuId - 1) % BATCH_SIZE);
  
  console.log(`🔐 VU ${vuId} assigned to batch ${batchNumber}, position ${positionInBatch}`);
  
  // Calculate delay to stagger batches (wait for previous batch to complete)
  const batchDelay = batchNumber * 3; // 3 seconds between batches
  const withinBatchDelay = positionInBatch * 0.1; // 100ms stagger within batch
  const totalDelay = batchDelay + withinBatchDelay;
  
  if (totalDelay > 0) {
    console.log(`⏱️ VU ${vuId} waiting ${totalDelay}s (batch delay: ${batchDelay}s + position delay: ${withinBatchDelay}s)`);
    sleep(totalDelay);
  }
  
  // Assign each VU to a specific user (round-robin distribution)
  const assignedUserIndex = (vuId - 1) % users.length;
  
  // Try assigned user first
  let testUser = users[assignedUserIndex];
  console.log(`🔐 VU ${vuId} attempting authentication with assigned user ${testUser.email}...`);
  
  let authResult = authenticateUser(testUser);
  
  if (authResult) {
    vuTokenCache.set(vuId, authResult);
    console.log(`✅ VU ${vuId} authenticated successfully with assigned user ${testUser.email} in batch ${batchNumber}`);
    return authResult;
  }
  
  console.log(`❌ VU ${vuId} failed with assigned user ${testUser.email}, trying all other users...`);
  
  // If assigned user failed, try all other users as fallback
  for (let i = 0; i < users.length; i++) {
    if (i === assignedUserIndex) continue; // Skip already tried assigned user
    
    testUser = users[i];
    console.log(`🔐 VU ${vuId} attempting authentication with fallback user ${testUser.email}...`);
    
    authResult = authenticateUser(testUser);
    
    if (authResult) {
      vuTokenCache.set(vuId, authResult);
      console.log(`✅ VU ${vuId} authenticated successfully with fallback user ${testUser.email} in batch ${batchNumber}`);
      return authResult;
    }
    
    console.log(`❌ VU ${vuId} authentication failed with fallback user ${testUser.email}, trying next user...`);
  }
  
  // If all users failed, return null instead of crashing VU
  console.log(`⚠️ VU ${vuId} failed to authenticate with any test user in batch ${batchNumber} - continuing without auth`);
  return null;
}

/**
 * Clear authentication cache (useful for testing or cleanup)
 */
export function clearAuthCache() {
  vuTokenCache.clear();
  console.log('🧹 Authentication cache cleared');
}