// Targets for the AutoGPT Platform load tests, picked by K6_ENVIRONMENT
// (default DEV). PREVIEW also needs PR_NUMBER.
//
// - BACKEND_URL: the backend's origin; /health lives here.
// - API_BASE_URL: the backend's REST root (BACKEND_URL + /api).
// - FRONTEND_URL: the Next.js app. Better Auth runs under /api/auth there,
//   and the backend verifies tokens against its /api/auth/jwks.
// - SIGNUP_ALLOWED: whether the authenticated tests may sign up throwaway
//   accounts when LOAD_TEST_ACCOUNTS is unset (see lib/auth.js).
function target(name, backendUrl, frontendUrl, signupAllowed) {
  return {
    NAME: name,
    BACKEND_URL: backendUrl,
    API_BASE_URL: `${backendUrl}/api`,
    FRONTEND_URL: frontendUrl,
    SIGNUP_ALLOWED: signupAllowed,
  };
}

export function getEnvironmentConfig() {
  const env = (__ENV.K6_ENVIRONMENT || "DEV").toUpperCase();
  switch (env) {
    case "LOCAL": {
      // k6.sh sets LOCAL_HOST so k6 inside Docker reaches the host's stack.
      const host = __ENV.LOCAL_HOST || "localhost";
      return target(
        "LOCAL",
        `http://${host}:8006`,
        `http://${host}:3000`,
        true,
      );
    }
    case "DEV":
      return target(
        "DEV",
        "https://dev-server.agpt.co",
        "https://dev-builder.agpt.co",
        false,
      );
    case "PREVIEW":
      return getPreviewConfig(__ENV.PR_NUMBER);
    case "PROD":
      if (__ENV.K6_ALLOW_PROD !== "true") {
        throw new Error(
          "K6_ENVIRONMENT=PROD sends load at production; agree it with the team, then set K6_ALLOW_PROD=true",
        );
      }
      return target(
        "PROD",
        "https://backend.agpt.co",
        "https://platform.agpt.co",
        false,
      );
    default:
      throw new Error(
        `Unknown K6_ENVIRONMENT "${env}": use PREVIEW, DEV, LOCAL or PROD`,
      );
  }
}

// K6_ENVIRONMENT=PREVIEW PR_NUMBER=<n> targets the `!deploy` preview of PR <n>.
function getPreviewConfig(prNumber) {
  if (!/^[1-9]\d{0,6}$/.test(prNumber || "")) {
    throw new Error(
      "K6_ENVIRONMENT=PREVIEW needs PR_NUMBER=<pull request number>",
    );
  }
  return target(
    `PREVIEW pr-${prNumber}`,
    `https://pr-${prNumber}-server.agpt.co`,
    `https://autogpt-pr-${prNumber}.vercel.app`,
    true,
  );
}

// Performance test configurations - Environment variable overrides supported
export const PERFORMANCE_CONFIG = {
  // Default load test parameters (override with env vars: VUS, DURATION, RAMP_UP, RAMP_DOWN)
  DEFAULT_VUS: parseInt(__ENV.VUS) || 10,
  DEFAULT_DURATION: __ENV.DURATION || "2m",
  DEFAULT_RAMP_UP: __ENV.RAMP_UP || "30s",
  DEFAULT_RAMP_DOWN: __ENV.RAMP_DOWN || "30s",

  // Stress test parameters (override with env vars: STRESS_VUS, STRESS_DURATION, etc.)
  STRESS_VUS: parseInt(__ENV.STRESS_VUS) || 50,
  STRESS_DURATION: __ENV.STRESS_DURATION || "5m",
  STRESS_RAMP_UP: __ENV.STRESS_RAMP_UP || "1m",
  STRESS_RAMP_DOWN: __ENV.STRESS_RAMP_DOWN || "1m",

  // Spike test parameters (override with env vars: SPIKE_VUS, SPIKE_DURATION, etc.)
  SPIKE_VUS: parseInt(__ENV.SPIKE_VUS) || 100,
  SPIKE_DURATION: __ENV.SPIKE_DURATION || "30s",
  SPIKE_RAMP_UP: __ENV.SPIKE_RAMP_UP || "10s",
  SPIKE_RAMP_DOWN: __ENV.SPIKE_RAMP_DOWN || "10s",

  // Volume test parameters (override with env vars: VOLUME_VUS, VOLUME_DURATION, etc.)
  VOLUME_VUS: parseInt(__ENV.VOLUME_VUS) || 20,
  VOLUME_DURATION: __ENV.VOLUME_DURATION || "10m",
  VOLUME_RAMP_UP: __ENV.VOLUME_RAMP_UP || "2m",
  VOLUME_RAMP_DOWN: __ENV.VOLUME_RAMP_DOWN || "2m",

  // SLA thresholds (adjustable via env vars: THRESHOLD_P95, THRESHOLD_P99, etc.)
  THRESHOLDS: {
    http_req_duration: [
      `p(95)<${__ENV.THRESHOLD_P95 || "2000"}`,
      `p(99)<${__ENV.THRESHOLD_P99 || "5000"}`,
    ],
    http_req_failed: [`rate<${__ENV.THRESHOLD_ERROR_RATE || "0.05"}`],
    http_reqs: [`rate>${__ENV.THRESHOLD_RPS || "10"}`],
    checks: [`rate>${__ENV.THRESHOLD_CHECK_RATE || "0.95"}`],
  },
};

// Helper function to get load test configuration based on test type
export function getLoadTestConfig(testType = "default") {
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
    },
  };

  return configs[testType] || configs.default;
}
