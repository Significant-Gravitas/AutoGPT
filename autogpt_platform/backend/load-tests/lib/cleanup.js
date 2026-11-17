// What teardown() removes of what the run created, through the routes the
// app deletes with: the runs of the seeded graphs (the seed's and the graph
// journey's), the library agents the seed created or added, the seeded chat
// sessions and expert chats, the chat sessions the journeys and probes
// started, and the experts the run hired, which it archives as a user does.
// The backend soft-deletes runs and library agents, as it does for a user,
// deletes chat sessions outright, and keeps an archived expert, its
// schedules paused.
//
// What stays is in the README (Seeding): the graphs, which the backend does
// not delete while a library entry refers to them, deleted or not; the
// archived experts and what their setup installed; the onboarding steps the
// seed completed and the credits they granted; the Stripe test-mode
// customers; and the throwaway accounts.
//
// A target that is briefly unavailable (no answer, or 502 to 504 from its
// ingress while a pod restarts) gets each request again, a few times and
// a little later each time, before the delete counts as failed.
import { sleep } from "k6";
import http from "k6/http";
import { Counter } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { accountCount, accountHeaders } from "./auth.js";
import { COVERED } from "./coverage.js";

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
const sessionsDeleted = new Counter("cleanup_sessions");
const expertsArchived = new Counter("cleanup_experts");
const cleanupFailures = new Counter("cleanup_failures");

// For options.teardownTimeout: the deletes go ten at a time, but an account
// can hold a few hundred runs after a stress run, and an archive stops the
// expert's sandbox.
export function cleanupTimeout() {
  return `${120 + accountCount() * 40}s`;
}

// `when` names the caller in the log: teardown, or a setup() that threw.
// An expert is archived after its chats are deleted.
export function removeRunData(data, when) {
  if (!data || !Array.isArray(data.accounts)) return;
  const done = {
    executions: 0,
    libraryAgents: 0,
    sessions: 0,
    experts: 0,
    failed: 0,
  };
  for (const account of data.accounts) {
    const headers = accountHeaders(data, account.index);
    const runs = [];
    // A listing that could not be read counts once: what it held stays.
    let unlisted = 0;
    for (const graph of account.graphs) {
      const listed = graphRuns(graph.graphId, headers);
      for (const id of listed.ids) runs.push(id);
      if (!listed.complete) unlisted++;
    }
    const sessionList = runSessions(account, data, headers);
    if (!sessionList.complete) unlisted++;
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
      sessionList.ids.map((id) => `/chat/sessions/${id}`),
      "/chat/sessions/{session_id}",
      headers,
    );
    const expertList = runExperts(account, headers);
    if (!expertList.complete) unlisted++;
    const experts = deleteAll(
      expertList.ids.map((id) => `/experts/${id}`),
      "/experts/{expert_id}",
      headers,
    );
    if (executions.deleted) executionsDeleted.add(executions.deleted);
    if (libraryAgents.deleted) libraryAgentsDeleted.add(libraryAgents.deleted);
    if (sessions.deleted) sessionsDeleted.add(sessions.deleted);
    if (experts.deleted) expertsArchived.add(experts.deleted);
    const failed =
      executions.failed +
      libraryAgents.failed +
      sessions.failed +
      experts.failed +
      unlisted;
    if (failed) cleanupFailures.add(failed);
    done.executions += executions.deleted;
    done.libraryAgents += libraryAgents.deleted;
    done.sessions += sessions.deleted;
    done.experts += experts.deleted;
    done.failed += failed;
  }
  console.log(
    `${when}: deleted ${done.executions} graph runs, ${done.libraryAgents} library agents and ${done.sessions} chat sessions, archived ${done.experts} experts${
      done.failed > 0
        ? `; ${done.failed} deletes or listings failed, the log above says which`
        : ""
    }; the graphs stay (README: Seeding)`,
  );
}

// The account's active experts the run hired: every one it did not have
// before the run (the seed's list, empty for an account the run signed up),
// which covers the seed's work expert and anything expert_hire left active.
// An account the seed read no experts for had none hired. `complete` is
// false when the list could not be read; the work expert is archived then
// all the same.
function runExperts(account, headers) {
  const experts = account.experts;
  if (!experts || !Array.isArray(experts.preexisting)) {
    return { ids: [], complete: true };
  }
  const keep = experts.preexisting;
  const listed = read("/experts", "/experts", headers);
  if (!Array.isArray(listed)) {
    const ids =
      experts.workId && keep.indexOf(experts.workId) < 0
        ? [experts.workId]
        : [];
    return { ids, complete: false };
  }
  return {
    ids: listed
      .map((expert) => expert.id)
      .filter((id) => id && keep.indexOf(id) < 0),
    complete: true,
  };
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

// The seeded sessions and expert chats and, when a journey that starts
// sessions ran, every session an account that could open chats started
// after the run's setup began: the turn journeys', the expert journeys' and
// the probes'. The list holds expert chats too and is sorted by last update,
// so it is read until it reaches sessions untouched since then.
//
// Only an account that exists for load tests is swept that way: one the run
// signed up, or one named for a load-test engine. Anyone could be using any
// other account in LOAD_TEST_ACCOUNTS at the same time, so the run deletes
// only the sessions it seeded there and leaves the ones its journeys started.
function runSessions(account, data, headers) {
  const ids = new Set(
    account.sessionIds.concat(
      account.experts ? account.experts.sessionIds : [],
    ),
  );
  const loadTestOnly =
    data.signedUp || account.mode === "dummy" || account.mode === "scripted";
  if (!data.startsSessions || account.chat !== COVERED || !loadTestOnly) {
    return { ids: Array.from(ids), complete: true };
  }
  const since = timestamp(data.startedAt);
  for (let page = 0; page < MAX_PAGES; page++) {
    const body = read(
      `/chat/sessions?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}&pinned_first=false`,
      "/chat/sessions",
      headers,
    );
    if (!body || !Array.isArray(body.sessions)) {
      return { ids: Array.from(ids), complete: false };
    }
    for (const session of body.sessions) {
      if (timestamp(session.created_at) >= since) ids.add(session.id);
    }
    const last = body.sessions[body.sessions.length - 1];
    if (
      body.sessions.length < PAGE_SIZE ||
      timestamp(last.updated_at) < since
    ) {
      break;
    }
  }
  return { ids: Array.from(ids), complete: true };
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
