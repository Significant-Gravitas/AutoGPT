/**
 * Setup Test Users
 * 
 * Creates test users for load testing if they don't exist
 */

import http from 'k6/http';
import { check } from 'k6';
import { getEnvironmentConfig } from './configs/environment.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [{ duration: '5s', target: 1 }],
};

export default function () {
  console.log('🔧 Setting up test users...');
  
  const testUsers = [
    { email: 'loadtest1@example.com', password: 'LoadTest123!' },
    { email: 'loadtest2@example.com', password: 'LoadTest123!' },
    { email: 'loadtest3@example.com', password: 'LoadTest123!' },
  ];
  
  for (const user of testUsers) {
    createTestUser(user.email, user.password);
  }
}

function createTestUser(email, password) {
  console.log(`👤 Creating user: ${email}`);
  
  const signupUrl = `${config.SUPABASE_URL}/auth/v1/signup`;
  
  const signupPayload = {
    email: email,
    password: password,
    data: {
      full_name: `Load Test User`,
      username: email.split('@')[0],
    }
  };
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'apikey': config.SUPABASE_ANON_KEY,
    },
  };
  
  const response = http.post(signupUrl, JSON.stringify(signupPayload), params);
  
  const success = check(response, {
    'User creation: Status is 200 or user exists': (r) => r.status === 200 || r.status === 422,
    'User creation: Response time < 3s': (r) => r.timings.duration < 3000,
  });
  
  if (response.status === 200) {
    console.log(`✅ Created user: ${email}`);
  } else if (response.status === 422) {
    console.log(`ℹ️  User already exists: ${email}`);
  } else {
    console.error(`❌ Failed to create user ${email}: ${response.status} - ${response.body}`);
  }
  
  return success;
}