// What teardown() removes of what the run created, through the routes the
// app deletes with: the runs of the seeded graphs (the seed's and the graph
// journey's), the library agents the seed created or added, the seeded chat
// sessions, and the chat sessions the turn journey started. The backend
// soft-deletes runs and library agents, as it does for a user, and deletes
// chat sessions outright.
//
// What stays is in the README (Seeding): the graphs, which the backend does
// not delete while a library entry refers to them, deleted or not; the
// onboarding steps the seed completed and the credits they granted; the
// Stripe test-mode customers; and the throwaway accounts.
import http from "k6/http";
import { Counter } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { accountCount, accountHeaders } from "./auth.js";

const config = getEnvironmentConfig();
const API = config.API_BASE_URL;
const BATCH = 10;
const PAGE_SIZE = 100;
const MAX_PAGES = 50;

const executionsDeleted = new Counter("cleanup_executions");
const libraryAgentsDeleted = new Counter("cleanup_library_agents");
const sessionsDeleted = new Counter("cleanup_sessions");
const deletesFailed = new Counter("cleanup_failures");

// For options.teardownTimeout: the deletes go ten at a time, but an account
// can hold a few hundred runs after a stress run.
export function cleanupTimeout() {
  return `${120 + accountCount() * 30}s`;
}

// `when` names the caller in the log: teardown, or a setup() that threw.
export function removeRunData(data, when) {
  if (!data || !Array.isArray(data.accounts)) return;
  const done = { executions: 0, libraryAgents: 0, sessions: 0, failed: 0 };
  for (const account of data.accounts) {
    const headers = accountHeaders(data, account.index);
    const runs = [];
    for (const graph of account.graphs) {
      for (const id of graphRuns(graph.graphId, headers)) runs.push(id);
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
    const sessions = deleteAll(
      runSessions(account, data, headers).map((id) => `/chat/sessions/${id}`),
      "/chat/sessions/{session_id}",
      headers,
    );
    if (executions.deleted) executionsDeleted.add(executions.deleted);
    if (libraryAgents.deleted) libraryAgentsDeleted.add(libraryAgents.deleted);
    if (sessions.deleted) sessionsDeleted.add(sessions.deleted);
    const failed = executions.failed + libraryAgents.failed + sessions.failed;
    if (failed) deletesFailed.add(failed);
    done.executions += executions.deleted;
    done.libraryAgents += libraryAgents.deleted;
    done.sessions += sessions.deleted;
    done.failed += failed;
  }
  console.log(
    `${when}: deleted ${done.executions} graph runs, ${done.libraryAgents} library agents and ${done.sessions} chat sessions${
      done.failed > 0
        ? `; ${done.failed} deletes failed, the log above says which`
        : ""
    }; the graphs stay (README: Seeding)`,
  );
}

// Every run of a graph the seed created: none existed before the run.
function graphRuns(graphId, headers) {
  const ids = [];
  for (let page = 1; page <= MAX_PAGES; page++) {
    const body = read(
      `/graphs/${graphId}/executions?page=${page}&page_size=${PAGE_SIZE}`,
      "/graphs/{graph_id}/executions",
      headers,
    );
    const executions =
      body && Array.isArray(body.executions) ? body.executions : [];
    for (const execution of executions) {
      if (execution.id) ids.push(execution.id);
    }
    const pages = body && body.pagination ? body.pagination.total_pages : 0;
    if (executions.length === 0 || page >= pages) break;
  }
  return ids;
}

// The seeded sessions and, when the turn journey ran, every session the
// account started after the run's setup began. The list is sorted by last
// update, so it is read until it reaches sessions untouched since then.
function runSessions(account, data, headers) {
  const ids = new Set(account.sessionIds);
  if (!data.copilot || !data.copilot.enabled) return Array.from(ids);
  const since = timestamp(data.startedAt);
  for (let page = 0; page < MAX_PAGES; page++) {
    const body = read(
      `/chat/sessions?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}&pinned_first=false`,
      "/chat/sessions",
      headers,
    );
    const sessions = body && Array.isArray(body.sessions) ? body.sessions : [];
    for (const session of sessions) {
      if (timestamp(session.created_at) >= since) ids.add(session.id);
    }
    const last = sessions[sessions.length - 1];
    if (sessions.length < PAGE_SIZE || timestamp(last.updated_at) < since) {
      break;
    }
  }
  return Array.from(ids);
}

// The backend's timestamps carry microseconds and an offset, or no zone for
// UTC; Date.parse is only specified for milliseconds.
function timestamp(text) {
  const match =
    /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?$/.exec(
      String(text || ""),
    );
  if (!match) return NaN;
  const fraction = (match[2] || ".").slice(0, 4).padEnd(4, "0");
  return Date.parse(`${match[1]}${fraction}${match[3] || "Z"}`);
}

function read(path, route, headers) {
  const res = http.get(`${API}${path}`, {
    headers,
    tags: { name: `/api${route}`, phase: "teardown" },
    responseCallback: http.expectedStatuses(200),
    timeout: "60s",
  });
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
  for (let i = 0; i < paths.length; i += BATCH) {
    const responses = http.batch(
      paths.slice(i, i + BATCH).map((path) => ({
        method: "DELETE",
        url: `${API}${path}`,
        params: {
          headers,
          tags: { name: `/api${route}`, phase: "teardown" },
          responseCallback: http.expectedStatuses(200, 204),
          timeout: "60s",
        },
      })),
    );
    for (const res of responses) {
      if (res.status === 200 || res.status === 204) {
        result.deleted++;
      } else {
        result.failed++;
        console.warn(
          `teardown: DELETE /api${route} answered HTTP ${res.status}`,
        );
      }
    }
  }
  return result;
}
