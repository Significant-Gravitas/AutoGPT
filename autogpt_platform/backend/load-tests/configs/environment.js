// Environment configuration for AutoGPT Platform load tests
export const ENV_CONFIG = {
  DEV: {
    API_BASE_URL: 'https://dev-server.agpt.co',
    BUILDER_BASE_URL: 'https://dev-builder.agpt.co', 
    WS_BASE_URL: 'wss://dev-ws-server.agpt.co',
    SUPABASE_URL: 'https://adfjtextkuilwuhzdjpf.supabase.co',
  },
  STAGING: {
    API_BASE_URL: 'https://staging-api.agpt.co',
    BUILDER_BASE_URL: 'https://staging-builder.agpt.co',
    WS_BASE_URL: 'wss://staging-ws-server.agpt.co',
    SUPABASE_URL: 'https://staging-supabase.agpt.co',
  },
  PROD: {
    API_BASE_URL: 'https://api.agpt.co',
    BUILDER_BASE_URL: 'https://builder.agpt.co',
    WS_BASE_URL: 'wss://ws-server.agpt.co',
    SUPABASE_URL: 'https://supabase.agpt.co',
  }
};

// Get environment config based on K6_ENVIRONMENT variable (default: DEV)
export function getEnvironmentConfig() {
  const env = __ENV.K6_ENVIRONMENT || 'DEV';
  return ENV_CONFIG[env];
}

// Authentication configuration
export const AUTH_CONFIG = {
  // Test user credentials - REPLACE WITH ACTUAL TEST ACCOUNTS
  TEST_USERS: [
    {
      email: 'loadtest1@example.com',
      password: 'LoadTest123!',
      user_id: 'test-user-1'
    },
    {
      email: 'loadtest2@example.com', 
      password: 'LoadTest123!',
      user_id: 'test-user-2'
    },
    {
      email: 'loadtest3@example.com',
      password: 'LoadTest123!', 
      user_id: 'test-user-3'
    }
  ],
  
  // JWT token for API access (will be set during test execution)
  JWT_TOKEN: null,
  
  // Service role key for admin operations (actual dev instance key)
  SERVICE_ROLE_KEY: __ENV.SUPABASE_SERVICE_ROLE_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFkZmp0ZXh0a3VpbHd1aHpkanBmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMDI1MTcwMiwiZXhwIjoyMDQ1ODI3NzAyfQ.JHadCgyuMVejDxl66DIe4ZlB1ra7IGDLEkABhSJm540'
};

// Performance test configurations - Environment variable overrides supported
export const PERFORMANCE_CONFIG = {
  // Default load test parameters (override with env vars: VUS, DURATION, RAMP_UP, RAMP_DOWN)
  DEFAULT_VUS: parseInt(__ENV.VUS) || 10,
  DEFAULT_DURATION: __ENV.DURATION || '2m',
  DEFAULT_RAMP_UP: __ENV.RAMP_UP || '30s',
  DEFAULT_RAMP_DOWN: __ENV.RAMP_DOWN || '30s',
  
  // Stress test parameters (override with env vars: STRESS_VUS, STRESS_DURATION, etc.)
  STRESS_VUS: parseInt(__ENV.STRESS_VUS) || 50,
  STRESS_DURATION: __ENV.STRESS_DURATION || '5m',
  STRESS_RAMP_UP: __ENV.STRESS_RAMP_UP || '1m',
  STRESS_RAMP_DOWN: __ENV.STRESS_RAMP_DOWN || '1m',
  
  // Spike test parameters (override with env vars: SPIKE_VUS, SPIKE_DURATION, etc.)
  SPIKE_VUS: parseInt(__ENV.SPIKE_VUS) || 100,
  SPIKE_DURATION: __ENV.SPIKE_DURATION || '30s',
  SPIKE_RAMP_UP: __ENV.SPIKE_RAMP_UP || '10s',
  SPIKE_RAMP_DOWN: __ENV.SPIKE_RAMP_DOWN || '10s',
  
  // Volume test parameters (override with env vars: VOLUME_VUS, VOLUME_DURATION, etc.)
  VOLUME_VUS: parseInt(__ENV.VOLUME_VUS) || 20,
  VOLUME_DURATION: __ENV.VOLUME_DURATION || '10m',
  VOLUME_RAMP_UP: __ENV.VOLUME_RAMP_UP || '2m',
  VOLUME_RAMP_DOWN: __ENV.VOLUME_RAMP_DOWN || '2m',
  
  // SLA thresholds (adjustable via env vars: THRESHOLD_P95, THRESHOLD_P99, etc.)
  THRESHOLDS: {
    http_req_duration: [
      `p(95)<${__ENV.THRESHOLD_P95 || '2000'}`, 
      `p(99)<${__ENV.THRESHOLD_P99 || '5000'}`
    ],
    http_req_failed: [`rate<${__ENV.THRESHOLD_ERROR_RATE || '0.05'}`], 
    http_reqs: [`rate>${__ENV.THRESHOLD_RPS || '10'}`], 
    checks: [`rate>${__ENV.THRESHOLD_CHECK_RATE || '0.95'}`], 
  }
};

// Helper function to get load test configuration based on test type
export function getLoadTestConfig(testType = 'default') {
  const configs = {
    default: {
      vus: PERFORMANCE_CONFIG.DEFAULT_VUS,
      duration: PERFORMANCE_CONFIG.DEFAULT_DURATION,
      rampUp: PERFORMANCE_CONFIG.DEFAULT_RAMP_UP,
      rampDown: PERFORMANCE_CONFIG.DEFAULT_RAMP_DOWN,
    },
    stress: {
      vus: PERFORMANCE_CONFIG.STRESS_VUS,
      duration: PERFORMANCE_CONFIG.STRESS_DURATION,
      rampUp: PERFORMANCE_CONFIG.STRESS_RAMP_UP,
      rampDown: PERFORMANCE_CONFIG.STRESS_RAMP_DOWN,
    },
    spike: {
      vus: PERFORMANCE_CONFIG.SPIKE_VUS,
      duration: PERFORMANCE_CONFIG.SPIKE_DURATION,
      rampUp: PERFORMANCE_CONFIG.SPIKE_RAMP_UP,
      rampDown: PERFORMANCE_CONFIG.SPIKE_RAMP_DOWN,
    },
    volume: {
      vus: PERFORMANCE_CONFIG.VOLUME_VUS,
      duration: PERFORMANCE_CONFIG.VOLUME_DURATION,
      rampUp: PERFORMANCE_CONFIG.VOLUME_RAMP_UP,
      rampDown: PERFORMANCE_CONFIG.VOLUME_RAMP_DOWN,
    }
  };
  
  return configs[testType] || configs.default;
}

// Grafana Cloud K6 configuration
export const GRAFANA_CONFIG = {
  PROJECT_ID: __ENV.K6_CLOUD_PROJECT_ID || '',
  TOKEN: __ENV.K6_CLOUD_TOKEN || '',
  // Tags for organizing test results
  TEST_TAGS: {
    team: 'platform',
    service: 'autogpt-platform', 
    environment: __ENV.K6_ENVIRONMENT || 'dev',
    version: __ENV.GIT_COMMIT || 'unknown'
  }
};