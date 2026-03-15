import { check } from "k6";
import http from "k6/http";
import { Counter } from "k6/metrics";

import { getEnvironmentConfig } from "../../configs/environment.js";
import { getPreAuthenticatedHeaders } from "../../configs/pre-authenticated-tokens.js";

const config = getEnvironmentConfig();
const BASE_URL = config.API_BASE_URL;

// Custom metrics
const libraryRequests = new Counter("library_requests_total");
const successfulRequests = new Counter("successful_requests_total");
const failedRequests = new Counter("failed_requests_total");
const authenticationAttempts = new Counter("authentication_attempts_total");
const authenticationSuccesses = new Counter("authentication_successes_total");

// Test configuration
const VUS = parseInt(__ENV.VUS) || 5;
const DURATION = __ENV.DURATION || "2m";
const RAMP_UP = __ENV.RAMP_UP || "30s";
const RAMP_DOWN = __ENV.RAMP_DOWN || "30s";
const REQUESTS_PER_VU = parseInt(__ENV.REQUESTS_PER_VU) || 5;

// Performance thresholds for authenticated endpoints
const THRESHOLD_P95 = parseInt(__ENV.THRESHOLD_P95) || 10000; // 10s for authenticated endpoints
const THRESHOLD_P99 = parseInt(__ENV.THRESHOLD_P99) || 20000; // 20s for authenticated endpoints
const THRESHOLD_ERROR_RATE = parseFloat(__ENV.THRESHOLD_ERROR_RATE) || 0.1; // 10% error rate
const THRESHOLD_CHECK_RATE = parseFloat(__ENV.THRESHOLD_CHECK_RATE) || 0.85; // 85% success rate

export const options = {
  stages: [
    { duration: RAMP_UP, target: VUS },
    { duration: DURATION, target: VUS },
    { duration: RAMP_DOWN, target: 0 },
  ],
  thresholds: {
    http_req_duration: [
      { threshold: `p(95)<${THRESHOLD_P95}`, abortOnFail: false },
      { threshold: `p(99)<${THRESHOLD_P99}`, abortOnFail: false },
    ],
    http_req_failed: [
      { threshold: `rate<${THRESHOLD_ERROR_RATE}`, abortOnFail: false },
    ],
    checks: [{ threshold: `rate>${THRESHOLD_CHECK_RATE}`, abortOnFail: false }],
  },
  tags: {
    test_type: "marketplace_library_authorized",
    environment: __ENV.K6_ENVIRONMENT || "DEV",
  },
};

export default function () {
  console.log(`📚 VU ${__VU} starting authenticated library journey...`);

  // Get pre-authenticated headers
  const headers = getPreAuthenticatedHeaders(__VU);
  if (!headers || !headers.Authorization) {
    console.log(`❌ VU ${__VU} authentication failed, skipping iteration`);
    authenticationAttempts.add(1);
    return;
  }

  authenticationAttempts.add(1);
  authenticationSuccesses.add(1);

  // Run multiple library operations per iteration
  for (let i = 0; i < REQUESTS_PER_VU; i++) {
    console.log(
      `🔄 VU ${__VU} starting library operation ${i + 1}/${REQUESTS_PER_VU}...`,
    );
    authenticatedLibraryJourney(headers);
  }
}

function authenticatedLibraryJourney(headers) {
  const journeyStart = Date.now();

  // Step 1: Get user's library agents
  console.log(`📖 VU ${__VU} fetching user library agents...`);
  const libraryAgentsResponse = http.get(
    `${BASE_URL}/api/library/agents?page=1&page_size=20`,
    { headers },
  );

  libraryRequests.add(1);
  const librarySuccess = check(libraryAgentsResponse, {
    "Library agents endpoint returns 200": (r) => r.status === 200,
    "Library agents response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Library agents response time < 10s": (r) => r.timings.duration < 10000,
  });

  if (librarySuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    console.log(
      `⚠️ VU ${__VU} library agents request failed: ${libraryAgentsResponse.status} - ${libraryAgentsResponse.body}`,
    );
  }

  // Step 2: Get favorite agents
  console.log(`⭐ VU ${__VU} fetching favorite library agents...`);
  const favoriteAgentsResponse = http.get(
    `${BASE_URL}/api/library/agents/favorites?page=1&page_size=10`,
    { headers },
  );

  libraryRequests.add(1);
  const favoritesSuccess = check(favoriteAgentsResponse, {
    "Favorite agents endpoint returns 200": (r) => r.status === 200,
    "Favorite agents response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents !== undefined && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Favorite agents response time < 10s": (r) => r.timings.duration < 10000,
  });

  if (favoritesSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    console.log(
      `⚠️ VU ${__VU} favorite agents request failed: ${favoriteAgentsResponse.status}`,
    );
  }

  // Step 3: Add marketplace agent to library (simulate discovering and adding an agent)
  console.log(`🛍️ VU ${__VU} browsing marketplace to add agent...`);

  // First get available store agents to find one to add
  const storeAgentsResponse = http.get(
    `${BASE_URL}/api/store/agents?page=1&page_size=5`,
  );

  libraryRequests.add(1);
  const storeAgentsSuccess = check(storeAgentsResponse, {
    "Store agents endpoint returns 200": (r) => r.status === 200,
    "Store agents response has data": (r) => {
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
  });

  if (storeAgentsSuccess) {
    successfulRequests.add(1);

    try {
      const storeAgentsJson = storeAgentsResponse.json();
      if (storeAgentsJson?.agents && storeAgentsJson.agents.length > 0) {
        const randomStoreAgent =
          storeAgentsJson.agents[
            Math.floor(Math.random() * storeAgentsJson.agents.length)
          ];

        if (randomStoreAgent?.store_listing_version_id) {
          console.log(
            `➕ VU ${__VU} adding agent "${randomStoreAgent.name || "Unknown"}" to library...`,
          );

          const addAgentPayload = {
            store_listing_version_id: randomStoreAgent.store_listing_version_id,
          };

          const addAgentResponse = http.post(
            `${BASE_URL}/api/library/agents`,
            JSON.stringify(addAgentPayload),
            { headers },
          );

          libraryRequests.add(1);
          const addAgentSuccess = check(addAgentResponse, {
            "Add agent returns 201 or 200 (created/already exists)": (r) =>
              r.status === 201 || r.status === 200,
            "Add agent response has id": (r) => {
              try {
                const json = r.json();
                return json && json.id;
              } catch {
                return false;
              }
            },
            "Add agent response time < 15s": (r) => r.timings.duration < 15000,
          });

          if (addAgentSuccess) {
            successfulRequests.add(1);

            // Step 4: Update the added agent (mark as favorite)
            try {
              const addedAgentJson = addAgentResponse.json();
              if (addedAgentJson?.id) {
                console.log(`⭐ VU ${__VU} marking agent as favorite...`);

                const updatePayload = {
                  is_favorite: true,
                  auto_update_version: true,
                };

                const updateAgentResponse = http.patch(
                  `${BASE_URL}/api/library/agents/${addedAgentJson.id}`,
                  JSON.stringify(updatePayload),
                  { headers },
                );

                libraryRequests.add(1);
                const updateSuccess = check(updateAgentResponse, {
                  "Update agent returns 200": (r) => r.status === 200,
                  "Update agent response has updated data": (r) => {
                    try {
                      const json = r.json();
                      return json && json.id && json.is_favorite === true;
                    } catch {
                      return false;
                    }
                  },
                  "Update agent response time < 10s": (r) =>
                    r.timings.duration < 10000,
                });

                if (updateSuccess) {
                  successfulRequests.add(1);
                } else {
                  failedRequests.add(1);
                  console.log(
                    `⚠️ VU ${__VU} update agent failed: ${updateAgentResponse.status}`,
                  );
                }

                // Step 5: Get specific library agent details
                console.log(`📄 VU ${__VU} fetching agent details...`);
                const agentDetailsResponse = http.get(
                  `${BASE_URL}/api/library/agents/${addedAgentJson.id}`,
                  { headers },
                );

                libraryRequests.add(1);
                const detailsSuccess = check(agentDetailsResponse, {
                  "Agent details returns 200": (r) => r.status === 200,
                  "Agent details response has complete data": (r) => {
                    try {
                      const json = r.json();
                      return json && json.id && json.name && json.graph_id;
                    } catch {
                      return false;
                    }
                  },
                  "Agent details response time < 10s": (r) =>
                    r.timings.duration < 10000,
                });

                if (detailsSuccess) {
                  successfulRequests.add(1);
                } else {
                  failedRequests.add(1);
                  console.log(
                    `⚠️ VU ${__VU} agent details failed: ${agentDetailsResponse.status}`,
                  );
                }

                // Step 6: Fork the library agent (simulate user customization)
                console.log(`🍴 VU ${__VU} forking agent for customization...`);
                const forkAgentResponse = http.post(
                  `${BASE_URL}/api/library/agents/${addedAgentJson.id}/fork`,
                  "",
                  { headers },
                );

                libraryRequests.add(1);
                const forkSuccess = check(forkAgentResponse, {
                  "Fork agent returns 200": (r) => r.status === 200,
                  "Fork agent response has new agent data": (r) => {
                    try {
                      const json = r.json();
                      return json && json.id && json.id !== addedAgentJson.id; // Should be different ID
                    } catch {
                      return false;
                    }
                  },
                  "Fork agent response time < 15s": (r) =>
                    r.timings.duration < 15000,
                });

                if (forkSuccess) {
                  successfulRequests.add(1);
                } else {
                  failedRequests.add(1);
                  console.log(
                    `⚠️ VU ${__VU} fork agent failed: ${forkAgentResponse.status}`,
                  );
                }
              }
            } catch (e) {
              console.warn(
                `⚠️ VU ${__VU} failed to parse added agent response: ${e}`,
              );
              failedRequests.add(1);
            }
          } else {
            failedRequests.add(1);
            console.log(
              `⚠️ VU ${__VU} add agent failed: ${addAgentResponse.status} - ${addAgentResponse.body}`,
            );
          }
        }
      }
    } catch (e) {
      console.warn(`⚠️ VU ${__VU} failed to parse store agents data: ${e}`);
      failedRequests.add(1);
    }
  } else {
    failedRequests.add(1);
    console.log(
      `⚠️ VU ${__VU} store agents request failed: ${storeAgentsResponse.status}`,
    );
  }

  // Step 7: Search library agents
  const searchTerms = ["automation", "api", "data", "social", "productivity"];
  const randomSearchTerm =
    searchTerms[Math.floor(Math.random() * searchTerms.length)];

  console.log(`🔍 VU ${__VU} searching library for "${randomSearchTerm}"...`);
  const searchLibraryResponse = http.get(
    `${BASE_URL}/api/library/agents?search_term=${encodeURIComponent(randomSearchTerm)}&page=1&page_size=10`,
    { headers },
  );

  libraryRequests.add(1);
  const searchLibrarySuccess = check(searchLibraryResponse, {
    "Search library returns 200": (r) => r.status === 200,
    "Search library response has data": (r) => {
      try {
        const json = r.json();
        return json && json.agents !== undefined && Array.isArray(json.agents);
      } catch {
        return false;
      }
    },
    "Search library response time < 10s": (r) => r.timings.duration < 10000,
  });

  if (searchLibrarySuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    console.log(
      `⚠️ VU ${__VU} search library failed: ${searchLibraryResponse.status}`,
    );
  }

  // Step 8: Get library agent by graph ID (simulate finding agent by backend graph)
  if (libraryAgentsResponse.status === 200) {
    try {
      const libraryJson = libraryAgentsResponse.json();
      if (libraryJson?.agents && libraryJson.agents.length > 0) {
        const randomLibraryAgent =
          libraryJson.agents[
            Math.floor(Math.random() * libraryJson.agents.length)
          ];

        if (randomLibraryAgent?.graph_id) {
          console.log(
            `🔗 VU ${__VU} fetching agent by graph ID "${randomLibraryAgent.graph_id}"...`,
          );
          const agentByGraphResponse = http.get(
            `${BASE_URL}/api/library/agents/by-graph/${randomLibraryAgent.graph_id}`,
            { headers },
          );

          libraryRequests.add(1);
          const agentByGraphSuccess = check(agentByGraphResponse, {
            "Agent by graph ID returns 200": (r) => r.status === 200,
            "Agent by graph response has data": (r) => {
              try {
                const json = r.json();
                return (
                  json &&
                  json.id &&
                  json.graph_id === randomLibraryAgent.graph_id
                );
              } catch {
                return false;
              }
            },
            "Agent by graph response time < 10s": (r) =>
              r.timings.duration < 10000,
          });

          if (agentByGraphSuccess) {
            successfulRequests.add(1);
          } else {
            failedRequests.add(1);
            console.log(
              `⚠️ VU ${__VU} agent by graph request failed: ${agentByGraphResponse.status}`,
            );
          }
        }
      }
    } catch (e) {
      console.warn(
        `⚠️ VU ${__VU} failed to parse library agents for graph lookup: ${e}`,
      );
      failedRequests.add(1);
    }
  }

  const journeyDuration = Date.now() - journeyStart;
  console.log(
    `✅ VU ${__VU} completed authenticated library journey in ${journeyDuration}ms`,
  );
}

export function handleSummary(data) {
  const summary = {
    test_type: "Marketplace Library Authorized Access Load Test",
    environment: __ENV.K6_ENVIRONMENT || "DEV",
    configuration: {
      virtual_users: VUS,
      duration: DURATION,
      ramp_up: RAMP_UP,
      ramp_down: RAMP_DOWN,
      requests_per_vu: REQUESTS_PER_VU,
    },
    performance_metrics: {
      total_requests: data.metrics.http_reqs?.count || 0,
      failed_requests: data.metrics.http_req_failed?.values?.passes || 0,
      avg_response_time: data.metrics.http_req_duration?.values?.avg || 0,
      p95_response_time: data.metrics.http_req_duration?.values?.p95 || 0,
      p99_response_time: data.metrics.http_req_duration?.values?.p99 || 0,
    },
    custom_metrics: {
      library_requests: data.metrics.library_requests_total?.values?.count || 0,
      successful_requests:
        data.metrics.successful_requests_total?.values?.count || 0,
      failed_requests: data.metrics.failed_requests_total?.values?.count || 0,
      authentication_attempts:
        data.metrics.authentication_attempts_total?.values?.count || 0,
      authentication_successes:
        data.metrics.authentication_successes_total?.values?.count || 0,
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
    authentication_metrics: {
      auth_success_rate:
        (data.metrics.authentication_successes_total?.values?.count || 0) /
        Math.max(
          1,
          data.metrics.authentication_attempts_total?.values?.count || 0,
        ),
    },
    user_journey_coverage: [
      "Authenticate with valid credentials",
      "Fetch user library agents",
      "Browse favorite library agents",
      "Discover marketplace agents",
      "Add marketplace agent to library",
      "Update agent preferences (favorites)",
      "View detailed agent information",
      "Fork agent for customization",
      "Search library agents by term",
      "Lookup agent by graph ID",
    ],
  };

  console.log("\n📚 MARKETPLACE LIBRARY AUTHORIZED TEST SUMMARY");
  console.log("==============================================");
  console.log(`Environment: ${summary.environment}`);
  console.log(`Virtual Users: ${summary.configuration.virtual_users}`);
  console.log(`Duration: ${summary.configuration.duration}`);
  console.log(`Requests per VU: ${summary.configuration.requests_per_vu}`);
  console.log(`Total Requests: ${summary.performance_metrics.total_requests}`);
  console.log(
    `Successful Requests: ${summary.custom_metrics.successful_requests}`,
  );
  console.log(`Failed Requests: ${summary.custom_metrics.failed_requests}`);
  console.log(
    `Auth Success Rate: ${Math.round(summary.authentication_metrics.auth_success_rate * 100)}%`,
  );
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
