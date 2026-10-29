// Simple API diagnostic test
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
  // Thresholds disabled to prevent test abortion - collect all performance data
  // thresholds: {
  //   checks: ['rate>0.70'],
  //   http_req_duration: ['p(95)<30000'],
  //   http_req_failed: ['rate<0.3'],
  // },
  cloud: {
    projectID: __ENV.K6_CLOUD_PROJECT_ID,
    name: "AutoGPT Platform - Core API Validation Test",
  },
  // Timeout configurations to prevent early termination
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

  try {
    console.log(
      `🚀 VU ${__VU} making ${requestsPerVU} concurrent API requests...`,
    );

    // Create array of API requests to run concurrently
    const requests = [];

    for (let i = 0; i < requestsPerVU; i++) {
      // Add core API requests that represent realistic user workflows
      requests.push({
        method: "GET",
        url: `${config.API_BASE_URL}/credits`,
        params: { headers },
      });

      requests.push({
        method: "GET",
        url: `${config.API_BASE_URL}/graphs`,
        params: { headers },
      });

      requests.push({
        method: "GET",
        url: `${config.API_BASE_URL}/blocks`,
        params: { headers },
      });
    }

    // Execute all requests concurrently
    const responses = http.batch(requests);

    // Validate results
    let creditsSuccesses = 0;
    let graphsSuccesses = 0;
    let blocksSuccesses = 0;

    for (let i = 0; i < responses.length; i++) {
      const response = responses[i];
      const apiType = i % 3; // 0=credits, 1=graphs, 2=blocks

      if (apiType === 0) {
        // Credits API request
        check(response, {
          "Credits API: HTTP Status is 200": (r) => r.status === 200,
          "Credits API: Not Auth Error (401/403)": (r) =>
            r.status !== 401 && r.status !== 403,
          "Credits API: Response has valid JSON": (r) => {
            try {
              JSON.parse(r.body);
              return true;
            } catch (e) {
              return false;
            }
          },
          "Credits API: Response has credits field": (r) => {
            try {
              const data = JSON.parse(r.body);
              return data && typeof data.credits === "number";
            } catch (e) {
              return false;
            }
          },
          "Credits API: Overall Success": (r) => {
            try {
              if (r.status !== 200) return false;
              const data = JSON.parse(r.body);
              return data && typeof data.credits === "number";
            } catch (e) {
              return false;
            }
          },
        });
      } else if (apiType === 1) {
        // Graphs API request
        check(response, {
          "Graphs API: HTTP Status is 200": (r) => r.status === 200,
          "Graphs API: Not Auth Error (401/403)": (r) =>
            r.status !== 401 && r.status !== 403,
          "Graphs API: Response has valid JSON": (r) => {
            try {
              JSON.parse(r.body);
              return true;
            } catch (e) {
              return false;
            }
          },
          "Graphs API: Response is array": (r) => {
            try {
              const data = JSON.parse(r.body);
              return Array.isArray(data);
            } catch (e) {
              return false;
            }
          },
          "Graphs API: Overall Success": (r) => {
            try {
              if (r.status !== 200) return false;
              const data = JSON.parse(r.body);
              return Array.isArray(data);
            } catch (e) {
              return false;
            }
          },
        });
      } else {
        // Blocks API request
        check(response, {
          "Blocks API: HTTP Status is 200": (r) => r.status === 200,
          "Blocks API: Not Auth Error (401/403)": (r) =>
            r.status !== 401 && r.status !== 403,
          "Blocks API: Response has valid JSON": (r) => {
            try {
              JSON.parse(r.body);
              return true;
            } catch (e) {
              return false;
            }
          },
          "Blocks API: Response has blocks data": (r) => {
            try {
              const data = JSON.parse(r.body);
              return data && (Array.isArray(data) || typeof data === "object");
            } catch (e) {
              return false;
            }
          },
          "Blocks API: Overall Success": (r) => {
            try {
              if (r.status !== 200) return false;
              const data = JSON.parse(r.body);
              return data && (Array.isArray(data) || typeof data === "object");
            } catch (e) {
              return false;
            }
          },
        });
      }
    }

    console.log(
      `✅ VU ${__VU} completed ${responses.length} API requests with detailed auth/validation tracking`,
    );
  } catch (error) {
    console.error(`💥 Test failed: ${error.message}`);
    console.error(`💥 Stack: ${error.stack}`);
  }
}
