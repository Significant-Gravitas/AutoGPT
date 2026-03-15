import { check } from "k6";
import http from "k6/http";
import { Counter } from "k6/metrics";

import { getEnvironmentConfig } from "../../configs/environment.js";

const config = getEnvironmentConfig();
const BASE_URL = config.API_BASE_URL;

// Custom metrics
const marketplaceRequests = new Counter("marketplace_requests_total");
const successfulRequests = new Counter("successful_requests_total");
const failedRequests = new Counter("failed_requests_total");

// HTTP error tracking
const httpErrors = new Counter("http_errors_by_status");

// Enhanced error logging function
function logHttpError(response, endpoint, method = "GET") {
  if (response.status !== 200) {
    console.error(
      `❌ VU ${__VU} ${method} ${endpoint} failed: status=${response.status}, error=${response.error || "unknown"}, body=${response.body ? response.body.substring(0, 200) : "empty"}`,
    );
    httpErrors.add(1, {
      status: response.status,
      endpoint: endpoint,
      method: method,
    });
  }
}

// Test configuration
const VUS = parseInt(__ENV.VUS) || 10;
const DURATION = __ENV.DURATION || "2m";
const RAMP_UP = __ENV.RAMP_UP || "30s";
const RAMP_DOWN = __ENV.RAMP_DOWN || "30s";

// Performance thresholds for marketplace browsing
const REQUEST_TIMEOUT = 60000; // 60s per request timeout
const THRESHOLD_P95 = parseInt(__ENV.THRESHOLD_P95) || 5000; // 5s for public endpoints
const THRESHOLD_P99 = parseInt(__ENV.THRESHOLD_P99) || 10000; // 10s for public endpoints
const THRESHOLD_ERROR_RATE = parseFloat(__ENV.THRESHOLD_ERROR_RATE) || 0.05; // 5% error rate
const THRESHOLD_CHECK_RATE = parseFloat(__ENV.THRESHOLD_CHECK_RATE) || 0.95; // 95% success rate

export const options = {
  stages: [
    { duration: RAMP_UP, target: VUS },
    { duration: DURATION, target: VUS },
    { duration: RAMP_DOWN, target: 0 },
  ],
  // Thresholds disabled to collect all results regardless of performance
  // thresholds: {
  //   http_req_duration: [
  //     { threshold: `p(95)<${THRESHOLD_P95}`, abortOnFail: false },
  //     { threshold: `p(99)<${THRESHOLD_P99}`, abortOnFail: false },
  //   ],
  //   http_req_failed: [{ threshold: `rate<${THRESHOLD_ERROR_RATE}`, abortOnFail: false }],
  //   checks: [{ threshold: `rate>${THRESHOLD_CHECK_RATE}`, abortOnFail: false }],
  // },
  tags: {
    test_type: "marketplace_public_access",
    environment: __ENV.K6_ENVIRONMENT || "DEV",
  },
};

export default function () {
  console.log(`🛒 VU ${__VU} starting marketplace browsing journey...`);

  // Simulate realistic user marketplace browsing journey
  marketplaceBrowsingJourney();
}

function marketplaceBrowsingJourney() {
  const journeyStart = Date.now();

  // Step 1: Browse marketplace homepage - get featured agents
  console.log(`🏪 VU ${__VU} browsing marketplace homepage...`);
  const featuredAgentsResponse = http.get(
    `${BASE_URL}/api/store/agents?featured=true&page=1&page_size=10`,
  );
  logHttpError(
    featuredAgentsResponse,
    "/api/store/agents?featured=true",
    "GET",
  );

  marketplaceRequests.add(1);
  const featuredSuccess = check(featuredAgentsResponse, {
    "Featured agents endpoint returns 200": (r) => r.status === 200,
    "Featured agents response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Featured agents responds within 60s": (r) =>
      r.timings.duration < REQUEST_TIMEOUT,
  });

  if (featuredSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 2: Browse all agents with pagination
  console.log(`📋 VU ${__VU} browsing all agents...`);
  const allAgentsResponse = http.get(
    `${BASE_URL}/api/store/agents?page=1&page_size=20`,
  );
  logHttpError(allAgentsResponse, "/api/store/agents", "GET");

  marketplaceRequests.add(1);
  const allAgentsSuccess = check(allAgentsResponse, {
    "All agents endpoint returns 200": (r) => r.status === 200,
    "All agents response has data": (r) => {
      try {
        const json = r.json();
        return (
          json &&
          json.agents &&
          Array.isArray(json.agents) &&
          json.agents.length > 0
        );
      } catch {
        return false;
      }
    },
    "All agents responds within 60s": (r) =>
      r.timings.duration < REQUEST_TIMEOUT,
  });

  if (allAgentsSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 3: Search for specific agents
  const searchQueries = [
    "automation",
    "social media",
    "data analysis",
    "productivity",
  ];
  const randomQuery =
    searchQueries[Math.floor(Math.random() * searchQueries.length)];

  console.log(`🔍 VU ${__VU} searching for "${randomQuery}" agents...`);
  const searchResponse = http.get(
    `${BASE_URL}/api/store/agents?search_query=${encodeURIComponent(randomQuery)}&page=1&page_size=10`,
  );
  logHttpError(searchResponse, "/api/store/agents (search)", "GET");

  marketplaceRequests.add(1);
  const searchSuccess = check(searchResponse, {
    "Search agents endpoint returns 200": (r) => r.status === 200,
    "Search agents response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Search agents responds within 60s": (r) =>
      r.timings.duration < REQUEST_TIMEOUT,
  });

  if (searchSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 4: Browse agents by category
  const categories = ["AI", "PRODUCTIVITY", "COMMUNICATION", "DATA", "SOCIAL"];
  const randomCategory =
    categories[Math.floor(Math.random() * categories.length)];

  console.log(`📂 VU ${__VU} browsing "${randomCategory}" category...`);
  const categoryResponse = http.get(
    `${BASE_URL}/api/store/agents?category=${randomCategory}&page=1&page_size=15`,
  );
  logHttpError(categoryResponse, "/api/store/agents (category)", "GET");

  marketplaceRequests.add(1);
  const categorySuccess = check(categoryResponse, {
    "Category agents endpoint returns 200": (r) => r.status === 200,
    "Category agents response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Category agents responds within 60s": (r) =>
      r.timings.duration < REQUEST_TIMEOUT,
  });

  if (categorySuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 5: Get specific agent details (simulate clicking on an agent)
  if (allAgentsResponse.status === 200) {
    try {
      const allAgentsJson = allAgentsResponse.json();
      if (allAgentsJson?.agents && allAgentsJson.agents.length > 0) {
        const randomAgent =
          allAgentsJson.agents[
            Math.floor(Math.random() * allAgentsJson.agents.length)
          ];

        if (randomAgent?.creator_username && randomAgent?.slug) {
          console.log(
            `📄 VU ${__VU} viewing agent details for "${randomAgent.slug}"...`,
          );
          const agentDetailsResponse = http.get(
            `${BASE_URL}/api/store/agents/${encodeURIComponent(randomAgent.creator_username)}/${encodeURIComponent(randomAgent.slug)}`,
          );
          logHttpError(
            agentDetailsResponse,
            "/api/store/agents/{creator}/{slug}",
            "GET",
          );

          marketplaceRequests.add(1);
          const agentDetailsSuccess = check(agentDetailsResponse, {
            "Agent details endpoint returns 200": (r) => r.status === 200,
            "Agent details response has data": (r) => {
              try {
                const json = r.json();
                return json && json.id && json.name && json.description;
              } catch {
                return false;
              }
            },
            "Agent details responds within 60s": (r) =>
              r.timings.duration < REQUEST_TIMEOUT,
          });

          if (agentDetailsSuccess) {
            successfulRequests.add(1);
          } else {
            failedRequests.add(1);
          }
        }
      }
    } catch (e) {
      console.warn(
        `⚠️ VU ${__VU} failed to parse agents data for details lookup: ${e}`,
      );
      failedRequests.add(1);
    }
  }

  // Step 6: Browse creators
  console.log(`👥 VU ${__VU} browsing creators...`);
  const creatorsResponse = http.get(
    `${BASE_URL}/api/store/creators?page=1&page_size=20`,
  );
  logHttpError(creatorsResponse, "/api/store/creators", "GET");

  marketplaceRequests.add(1);
  const creatorsSuccess = check(creatorsResponse, {
    "Creators endpoint returns 200": (r) => r.status === 200,
    "Creators response has data": (r) => {
      try {
        const json = r.json();
        return json && json.creators && Array.isArray(json.creators);
      } catch {
        return false;
      }
    },
    "Creators responds within 60s": (r) => r.timings.duration < REQUEST_TIMEOUT,
  });

  if (creatorsSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 7: Get featured creators
  console.log(`⭐ VU ${__VU} browsing featured creators...`);
  const featuredCreatorsResponse = http.get(
    `${BASE_URL}/api/store/creators?featured=true&page=1&page_size=10`,
  );
  logHttpError(
    featuredCreatorsResponse,
    "/api/store/creators?featured=true",
    "GET",
  );

  marketplaceRequests.add(1);
  const featuredCreatorsSuccess = check(featuredCreatorsResponse, {
    "Featured creators endpoint returns 200": (r) => r.status === 200,
    "Featured creators response has data": (r) => {
      try {
        const json = r.json();
        return json && json.creators && Array.isArray(json.creators);
      } catch {
        return false;
      }
    },
    "Featured creators responds within 60s": (r) =>
      r.timings.duration < REQUEST_TIMEOUT,
  });

  if (featuredCreatorsSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
  }

  // Step 8: Get specific creator details (simulate clicking on a creator)
  if (creatorsResponse.status === 200) {
    try {
      const creatorsJson = creatorsResponse.json();
      if (creatorsJson?.creators && creatorsJson.creators.length > 0) {
        const randomCreator =
          creatorsJson.creators[
            Math.floor(Math.random() * creatorsJson.creators.length)
          ];

        if (randomCreator?.username) {
          console.log(
            `👤 VU ${__VU} viewing creator details for "${randomCreator.username}"...`,
          );
          const creatorDetailsResponse = http.get(
            `${BASE_URL}/api/store/creator/${encodeURIComponent(randomCreator.username)}`,
          );
          logHttpError(
            creatorDetailsResponse,
            "/api/store/creator/{username}",
            "GET",
          );

          marketplaceRequests.add(1);
          const creatorDetailsSuccess = check(creatorDetailsResponse, {
            "Creator details endpoint returns 200": (r) => r.status === 200,
            "Creator details response has data": (r) => {
              try {
                const json = r.json();
                return json && json.username && json.description !== undefined;
              } catch {
                return false;
              }
            },
            "Creator details responds within 60s": (r) =>
              r.timings.duration < REQUEST_TIMEOUT,
          });

          if (creatorDetailsSuccess) {
            successfulRequests.add(1);
          } else {
            failedRequests.add(1);
          }
        }
      }
    } catch (e) {
      console.warn(
        `⚠️ VU ${__VU} failed to parse creators data for details lookup: ${e}`,
      );
      failedRequests.add(1);
    }
  }

  const journeyDuration = Date.now() - journeyStart;
  console.log(
    `✅ VU ${__VU} completed marketplace browsing journey in ${journeyDuration}ms`,
  );
}

export function handleSummary(data) {
  const summary = {
    test_type: "Marketplace Public Access Load Test",
    environment: __ENV.K6_ENVIRONMENT || "DEV",
    configuration: {
      virtual_users: VUS,
      duration: DURATION,
      ramp_up: RAMP_UP,
      ramp_down: RAMP_DOWN,
    },
    performance_metrics: {
      total_requests: data.metrics.http_reqs?.count || 0,
      failed_requests: data.metrics.http_req_failed?.values?.passes || 0,
      avg_response_time: data.metrics.http_req_duration?.values?.avg || 0,
      p95_response_time: data.metrics.http_req_duration?.values?.p95 || 0,
      p99_response_time: data.metrics.http_req_duration?.values?.p99 || 0,
    },
    custom_metrics: {
      marketplace_requests:
        data.metrics.marketplace_requests_total?.values?.count || 0,
      successful_requests:
        data.metrics.successful_requests_total?.values?.count || 0,
      failed_requests: data.metrics.failed_requests_total?.values?.count || 0,
    },
    thresholds_met: {
      p95_threshold:
        (data.metrics.http_req_duration?.values?.p95 || 0) < THRESHOLD_P95,
      p99_threshold:
        (data.metrics.http_req_duration?.values?.p99 || 0) < THRESHOLD_P99,
      error_rate_threshold:
        (data.metrics.http_req_failed?.values?.rate || 0) <
        THRESHOLD_ERROR_RATE,
      check_rate_threshold:
        (data.metrics.checks?.values?.rate || 0) > THRESHOLD_CHECK_RATE,
    },
    user_journey_coverage: [
      "Browse featured agents",
      "Browse all agents with pagination",
      "Search agents by keywords",
      "Filter agents by category",
      "View specific agent details",
      "Browse creators directory",
      "View featured creators",
      "View specific creator details",
    ],
  };

  console.log("\n📊 MARKETPLACE PUBLIC ACCESS TEST SUMMARY");
  console.log("==========================================");
  console.log(`Environment: ${summary.environment}`);
  console.log(`Virtual Users: ${summary.configuration.virtual_users}`);
  console.log(`Duration: ${summary.configuration.duration}`);
  console.log(`Total Requests: ${summary.performance_metrics.total_requests}`);
  console.log(
    `Successful Requests: ${summary.custom_metrics.successful_requests}`,
  );
  console.log(`Failed Requests: ${summary.custom_metrics.failed_requests}`);
  console.log(
    `Average Response Time: ${Math.round(summary.performance_metrics.avg_response_time)}ms`,
  );
  console.log(
    `95th Percentile: ${Math.round(summary.performance_metrics.p95_response_time)}ms`,
  );
  console.log(
    `99th Percentile: ${Math.round(summary.performance_metrics.p99_response_time)}ms`,
  );

  console.log("\n🎯 Threshold Status:");
  console.log(
    `P95 < ${THRESHOLD_P95}ms: ${summary.thresholds_met.p95_threshold ? "✅" : "❌"}`,
  );
  console.log(
    `P99 < ${THRESHOLD_P99}ms: ${summary.thresholds_met.p99_threshold ? "✅" : "❌"}`,
  );
  console.log(
    `Error Rate < ${THRESHOLD_ERROR_RATE * 100}%: ${summary.thresholds_met.error_rate_threshold ? "✅" : "❌"}`,
  );
  console.log(
    `Check Rate > ${THRESHOLD_CHECK_RATE * 100}%: ${summary.thresholds_met.check_rate_threshold ? "✅" : "❌"}`,
  );

  return {
    stdout: JSON.stringify(summary, null, 2),
  };
}
