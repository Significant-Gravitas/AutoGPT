// What teardown() removes of what the run created, through the routes the
// app deletes with: the runs of the seeded graphs (the seed's and the graph
// journey's) and the library agents the seed created or added. The backend
// soft-deletes both, as it does for a user.
//
// What stays is in the README (Seeding): the graphs, which the backend does
// not delete while a library entry refers to them, deleted or not; the chat
// sessions, since deleting one also starts the copilot's browser cleanup
// for it on the backend, work a teardown should not add to the target it
// has just measured; the onboarding steps the seed completed and the
// credits they granted; the Stripe test-mode customers; and the throwaway
// accounts.
//
// A target that is briefly unavailable (no answer, or 502 to 504 from its
// ingress while a pod restarts) gets each request again, a few times and
// a little later each time, before the delete counts as failed.
import { sleep } from "k6";
import http from "k6/http";
import { Counter } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { accountCount, accountHeaders } from "./auth.js";

const config = getEnvironmentConfig();
const API = config.API_BASE_URL;
const BATCH = 10;
const PAGE_SIZE = 100;
const MAX_PAGES = 50;
const TRANSIENT = [0, 502, 503, 504];
const RETRIES = 4;
const RETRY_PAUSE_SECONDS = 5;

const executionsDeleted = new Counter("cleanup_executions");
const libraryAgentsDeleted = new Counter("cleanup_library_agents");
const cleanupFailures = new Counter("cleanup_failures");

// For options.teardownTimeout: the deletes go ten at a time, but an account
// can hold a few hundred runs after a stress run.
export function cleanupTimeout() {
  return `${120 + accountCount() * 30}s`;
}

// `when` names the caller in the log: teardown, or a setup() that threw.
export function removeRunData(data, when) {
  if (!data || !Array.isArray(data.accounts)) return;
  const done = { executions: 0, libraryAgents: 0, failed: 0 };
  for (const account of data.accounts) {
    const headers = accountHeaders(data, account.index);
    const runs = [];
    let unlisted = 0;
    for (const graph of account.graphs) {
      const listed = graphRuns(graph.graphId, headers);
      for (const id of listed.ids) runs.push(id);
      if (!listed.complete) unlisted++;
    }
    const executions = deleteAll(
      runs.map((id) => `/executions/${id}`),
      "/executions/{graph_exec_id}",
      headers,
    );
    const libraryAgents = deleteAll(
      account.libraryAgentIds.map((id) => `/library/agents/${id}`),
      "/library/agents/{library_agent_id}",
      headers,
    );
    if (executions.deleted) executionsDeleted.add(executions.deleted);
    if (libraryAgents.deleted) libraryAgentsDeleted.add(libraryAgents.deleted);
    // A graph whose runs could not be listed counts once: its runs stay.
    const failed = executions.failed + libraryAgents.failed + unlisted;
    if (failed) cleanupFailures.add(failed);
    done.executions += executions.deleted;
    done.libraryAgents += libraryAgents.deleted;
    done.failed += failed;
  }
  console.log(
    `${when}: deleted ${done.executions} graph runs and ${done.libraryAgents} library agents${
      done.failed > 0
        ? `; ${done.failed} deletes or listings failed, the log above says which`
        : ""
    }; the graphs and chat sessions stay (README: Seeding)`,
  );
}

// Every run of a graph the seed created: none existed before the run.
// `complete` is false when a page could not be read, so some may be missing.
function graphRuns(graphId, headers) {
  const ids = [];
  for (let page = 1; page <= MAX_PAGES; page++) {
    const body = read(
      `/graphs/${graphId}/executions?page=${page}&page_size=${PAGE_SIZE}`,
      "/graphs/{graph_id}/executions",
      headers,
    );
    if (!body || !Array.isArray(body.executions)) {
      return { ids, complete: false };
    }
    for (const execution of body.executions) {
      if (execution.id) ids.push(execution.id);
    }
    const pages = body.pagination ? body.pagination.total_pages : 0;
    if (body.executions.length === 0 || page >= pages) break;
  }
  return { ids, complete: true };
}

function params(route, headers, expected) {
  return {
    headers,
    tags: { name: `/api${route}`, phase: "teardown" },
    responseCallback: http.expectedStatuses(...expected),
    timeout: "60s",
  };
}

function read(path, route, headers) {
  let res;
  for (let attempt = 0; ; attempt++) {
    res = http.get(`${API}${path}`, params(route, headers, [200]));
    if (!TRANSIENT.includes(res.status) || attempt >= RETRIES) break;
    sleep(RETRY_PAUSE_SECONDS * (attempt + 1));
  }
  if (res.status !== 200) {
    console.warn(`teardown: GET /api${route} answered HTTP ${res.status}`);
    return null;
  }
  try {
    return res.json();
  } catch (e) {
    return null;
  }
}

function deleteAll(paths, route, headers) {
  const result = { deleted: 0, failed: 0 };
  let pending = paths;
  for (let attempt = 0; pending.length > 0; attempt++) {
    if (attempt > 0) sleep(RETRY_PAUSE_SECONDS * attempt);
    const again = [];
    for (let i = 0; i < pending.length; i += BATCH) {
      const batch = pending.slice(i, i + BATCH);
      const responses = http.batch(
        batch.map((path) => ({
          method: "DELETE",
          url: `${API}${path}`,
          params: params(route, headers, [200, 204]),
        })),
      );
      responses.forEach((res, j) => {
        // A 404 on a second attempt means the first went through and only
        // its answer was lost.
        if (
          res.status === 200 ||
          res.status === 204 ||
          (res.status === 404 && attempt > 0)
        ) {
          result.deleted++;
        } else if (TRANSIENT.includes(res.status) && attempt < RETRIES) {
          again.push(batch[j]);
        } else {
          result.failed++;
          console.warn(
            `teardown: DELETE /api${route} answered HTTP ${res.status}`,
          );
        }
      });
    }
    pending = again;
  }
  return result;
}
