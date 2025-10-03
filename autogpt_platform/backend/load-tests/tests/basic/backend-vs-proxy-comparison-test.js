// Comparison test: Direct backend vs Frontend proxy performance
import http from "k6/http";
import { check } from "k6";
import { getEnvironmentConfig } from "../../configs/environment.js";
import { getPreAuthenticatedHeaders } from "../../configs/pre-authenticated-tokens.js";

const config = getEnvironmentConfig();

export const options = {
  stages: [
    { duration: __ENV.RAMP_UP || "10s", target: parseInt(__ENV.VUS) || 3 },
    { duration: __ENV.DURATION || "30s", target: parseInt(__ENV.VUS) || 3 },
    { duration: __ENV.RAMP_DOWN || "10s", target: 0 },
  ],
  thresholds: {
    checks: ["rate>0.50"],
    http_req_duration: ["p(95)<60000"],
    http_req_failed: ["rate<0.5"],
    // Separate thresholds for each test type
    "http_req_duration{test_type:direct}": ["p(95)<30000"],
    "http_req_duration{test_type:proxy}": ["p(95)<60000"], // Allow more time for proxy
  },
  cloud: {
    projectID: parseInt(__ENV.K6_CLOUD_PROJECT_ID) || 4254406,
    name: `AutoGPT Backend vs Proxy Comparison - ${__ENV.ENDPOINT || "credits"} API`,
  },
};

function getProxyHeaders(vuId) {
  // Use the same user authentication tokens as direct backend access
  const userHeaders = getPreAuthenticatedHeaders(vuId);
  return {
    "Authorization": userHeaders.Authorization,
    "Content-Type": "application/json",
    "X-Client-Info": "k6-comparison-test"
  };
}

export default function () {
  const endpoint = __ENV.ENDPOINT || "credits";
  const testBoth = __ENV.TEST_MODE !== "proxy_only" && __ENV.TEST_MODE !== "direct_only";
  
  // URLs for both test types
  const directUrl = `${config.API_BASE_URL}/api/${endpoint}`;
  const proxyUrl = `${config.BUILDER_BASE_URL}/api/proxy/api/${endpoint}`;

  try {
    // Test 1: Direct Backend Access (if enabled)
    if (testBoth || __ENV.TEST_MODE === "direct_only") {
      const directHeaders = getPreAuthenticatedHeaders(__VU);
      
      if (directHeaders && directHeaders.Authorization) {
        const directResponse = http.get(directUrl, { 
          headers: directHeaders,
          tags: { test_type: "direct", endpoint: endpoint }
        });

        const directSuccess = check(directResponse, {
          [`${endpoint} Direct: Status is 200`]: (r) => r.status === 200,
          [`${endpoint} Direct: Response time < 3s`]: (r) => r.timings.duration < 3000,
        });

        console.log(
          `🔗 VU ${__VU} DIRECT /api/${endpoint}: ${directResponse.status} in ${directResponse.timings.duration}ms`
        );
      } else {
        console.log(`⚠️ VU ${__VU} DIRECT: No auth token available`);
      }
    }

    // Test 2: Frontend Proxy Access (if enabled)
    if (testBoth || __ENV.TEST_MODE === "proxy_only") {
      const proxyHeaders = getProxyHeaders(__VU);
      
      if (proxyHeaders && proxyHeaders.Authorization) {
        const proxyResponse = http.get(proxyUrl, { 
          headers: proxyHeaders,
          tags: { test_type: "proxy", endpoint: endpoint }
        });

        const proxySuccess = check(proxyResponse, {
          [`${endpoint} Proxy: Status is 200`]: (r) => r.status === 200,
          [`${endpoint} Proxy: Response time < 5s`]: (r) => r.timings.duration < 5000,
        });

        console.log(
          `🔄 VU ${__VU} PROXY /api/proxy/api/${endpoint}: ${proxyResponse.status} in ${proxyResponse.timings.duration}ms`
        );
      } else {
        console.log(`⚠️ VU ${__VU} PROXY: No auth token available`);
      }
    }

    // Performance comparison logging (only when testing both)
    if (testBoth) {
      console.log(`📊 VU ${__VU} completed ${endpoint} comparison test`);
    }

  } catch (error) {
    console.error(`💥 VU ${__VU} comparison test error: ${error.message}`);
  }
}