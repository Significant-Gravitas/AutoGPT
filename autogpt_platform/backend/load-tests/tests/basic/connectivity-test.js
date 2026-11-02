/**
 * Basic Connectivity Test
 *
 * Checks the pieces every authenticated test depends on: the backend answers
 * /health, the frontend serves the JWKS the backend verifies tokens against,
 * and the backend accepts a token minted by that frontend.
 */

import http from "k6/http";
import { check } from "k6";
import { getEnvironmentConfig } from "../../configs/environment.js";
import { authHeaders, authSetupTimeout, setupAuth } from "../../lib/auth.js";

const config = getEnvironmentConfig();

export const options = {
  stages: [
    { duration: __ENV.RAMP_UP || "1m", target: parseInt(__ENV.VUS) || 1 },
    { duration: __ENV.DURATION || "5m", target: parseInt(__ENV.VUS) || 1 },
    { duration: __ENV.RAMP_DOWN || "1m", target: 0 },
  ],
  thresholds: {
    checks: ["rate>0.70"],
    http_req_duration: ["p(95)<30000"],
    http_req_failed: ["rate<0.6"],
  },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: "AutoGPT Platform - Basic Connectivity & Auth Test",
  },
  setupTimeout: authSetupTimeout(),
  teardownTimeout: "60s",
  noConnectionReuse: false,
  userAgent: "k6-load-test/1.0",
};

export function setup() {
  return setupAuth();
}

export default function (data) {
  // Get load multiplier - how many concurrent requests each VU should make
  const requestsPerVU = parseInt(__ENV.REQUESTS_PER_VU) || 1;
  const headers = authHeaders(data);

  const requests = [];
  for (let i = 0; i < requestsPerVU; i++) {
    requests.push({ method: "GET", url: `${config.BACKEND_URL}/health` });
    requests.push({
      method: "GET",
      url: `${config.FRONTEND_URL}/api/auth/jwks`,
    });
    requests.push({
      method: "GET",
      url: `${config.API_BASE_URL}/credits`,
      params: { headers },
    });
  }

  const responses = http.batch(requests);

  for (let i = 0; i < responses.length; i++) {
    const response = responses[i];
    switch (i % 3) {
      case 0:
        check(response, {
          "Backend health: Status is 200": (r) => r.status === 200,
          "Backend health: Response time < 5s": (r) =>
            r.timings.duration < 5000,
        });
        break;
      case 1:
        check(response, {
          "Frontend JWKS: Status is 200": (r) => r.status === 200,
          "Frontend JWKS: Has signing keys": (r) => {
            try {
              const keys = r.json("keys");
              return Array.isArray(keys) && keys.length > 0;
            } catch (e) {
              return false;
            }
          },
        });
        break;
      default:
        check(response, {
          "Authenticated request: Status is 200": (r) => r.status === 200,
          "Authenticated request: Not Auth Error (401/403)": (r) =>
            r.status !== 401 && r.status !== 403,
        });
    }
  }
}

export function teardown() {
  console.log("Basic connectivity test completed");
}
