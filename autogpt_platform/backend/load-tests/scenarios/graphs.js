// A graph run the way the library page watches one: open the WebSocket,
// subscribe to the graph's executions, start a run, follow its events to a
// terminal status, then confirm with the execution's detail and list the
// graph's executions.
//
// Measured: ws_connect_ms (the socket opens), graph_run_start_ms (execute
// to the first RUNNING event), graph_run_ms (execute to COMPLETED),
// ws_event_latency_ms (an event's own ended_at to its arrival, so it
// includes any clock skew between the target and this machine).
//
// A run with no terminal event within GRAPH_RUN_TIMEOUT_MS is polled, for
// the rest of that timeout and at least GRAPH_RUN_POLL_MS more, so a run
// that finishes just after it is recorded as completed with its real
// duration and one that never finishes counts as failed. A socket that
// fails (it does not open, the server refuses the subscription or drops the
// connection) counts in ws_errors, and the run goes on by polling, tagged
// source:poll, so the executor is still measured.
//
// The socket URL's query carries the account's token, as the browser's
// does. The connection is tagged with the URL without its query (k6 uses
// the name tag as the connection's url tag), and errors are logged without
// it.
import exec from "k6/execution";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { setTimeout, clearTimeout } from "k6/timers";
import { WebSocket } from "k6/websockets";

import { getEnvironmentConfig } from "../configs/environment.js";
import { iterationPhase } from "../configs/profiles.js";
import { authHeaders } from "../lib/auth.js";
import { COVERED } from "../lib/coverage.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick } from "../lib/random.js";

const config = getEnvironmentConfig();

export const graphRunStart = new Trend("graph_run_start_ms", true);
export const graphRunTime = new Trend("graph_run_ms", true);
export const wsConnect = new Trend("ws_connect_ms", true);
export const wsEventLatency = new Trend("ws_event_latency_ms", true);
export const wsEvents = new Counter("ws_events");
export const wsErrors = new Rate("ws_errors");
export const graphRuns = new Counter("graph_runs");
export const graphRunsFailed = new Counter("graph_runs_failed");
export const graphRunErrors = new Rate("graph_run_errors");

const TERMINAL = ["COMPLETED", "FAILED", "TERMINATED"];
const RUN_TIMEOUT_MS = milliseconds("GRAPH_RUN_TIMEOUT_MS", 60000);
const POLL_GRACE_MS = milliseconds("GRAPH_RUN_POLL_MS", 30000);
const WS_OPEN_TIMEOUT_MS = 10000;
const WS_NAME = config.WS_URL.split("?")[0];

function milliseconds(variable, fallback) {
  const raw = __ENV[variable];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1000) {
    throw new Error(
      `${variable} must be a whole number of milliseconds, at least 1000; got "${raw}"`,
    );
  }
  return n;
}

// Error text can quote the socket URL, query and all.
function withoutToken(text) {
  return String(text).replace(/([?&]token=)[^&\s"']*/gi, "$1...");
}

export function graphRun(data, profile) {
  const phase = iterationPhase(profile);
  if (!data.graphs.enabled) {
    // Once per run: the scenario's first iteration, whichever VU runs it.
    if (exec.scenario.iterationInTest === 0) {
      console.warn(`graph_run: skipped: ${data.graphs.reason}`);
    }
    return;
  }
  const account = data.accounts[(__VU - 1) % data.accounts.length];
  // An account the seed could not start a run for sits the journey out;
  // the summary's coverage says how many accounts ran it.
  if (account.graphRuns !== COVERED) return;
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );
  const graph = pick(random, account.graphs);
  if (!graph) return;

  // The page shows the graph before it runs it.
  api("GET", `/graphs/${graph.graphId}`, {
    data,
    route: "/graphs/{graph_id}",
    endpoint: "graphs.graph_detail",
    phase,
  });

  const tags = { phase, endpoint: "graphs.run" };
  const run = {
    execId: null,
    executedAt: null,
    startedAt: null,
    endedAt: null,
    status: null,
    source: "ws",
    error: null,
    socketFailed: false,
  };

  const token = authHeaders(data).Authorization.replace(/^Bearer /, "");
  const connectAt = Date.now();
  let socket = null;
  try {
    socket = new WebSocket(
      `${config.WS_URL}?token=${encodeURIComponent(token)}`,
      null,
      { tags: { name: WS_NAME, endpoint: "graphs.ws", phase } },
    );
  } catch (e) {
    socketFailed(run, `the websocket could not be created: ${e}`);
  }
  if (socket === null) {
    finishRun(data, graph, run, phase, tags);
    return;
  }

  let opened = false;
  let closing = false;
  // Until the server acknowledges the subscription.
  let timer = setTimeout(() => {
    socketFailed(
      run,
      opened
        ? `the subscription was not acknowledged within ${WS_OPEN_TIMEOUT_MS}ms`
        : `the websocket did not open within ${WS_OPEN_TIMEOUT_MS}ms`,
    );
    close();
  }, WS_OPEN_TIMEOUT_MS);

  function close() {
    closing = true;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    try {
      socket.close();
    } catch (e) {
      // Already closed.
    }
  }

  socket.onopen = () => {
    opened = true;
    wsConnect.add(Date.now() - connectAt, tags);
    socket.send(
      JSON.stringify({
        method: "subscribe_graph_executions",
        data: { graph_id: graph.graphId },
      }),
    );
  };

  socket.onerror = (e) => {
    socketFailed(run, `websocket error: ${e && e.error ? e.error : e}`);
    close();
  };

  // The iteration ends when the socket has closed and no timer is pending,
  // so the reads that follow the run happen from here.
  socket.onclose = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (!closing) socketFailed(run, "the server closed the websocket");
    finishRun(data, graph, run, phase, tags);
  };

  socket.onmessage = (message) => {
    let frame;
    try {
      frame = JSON.parse(message.data);
    } catch (e) {
      return;
    }
    if (run.execId === null) {
      // The server acknowledges the subscription with a frame of the same
      // method, or refuses it with an error frame.
      if (frame.method === "error") {
        socketFailed(
          run,
          `the server refused the subscription: ${frame.error || "no reason given"}`,
        );
        close();
        return;
      }
      if (frame.method !== "subscribe_graph_executions" || !frame.success) {
        return;
      }
      clearTimeout(timer);
      timer = null;
      // Subscribed before the run exists, so no event is missed.
      startRun(data, graph, run, phase);
      if (run.execId === null) {
        close();
        return;
      }
      timer = setTimeout(() => {
        run.error = `no terminal event within ${RUN_TIMEOUT_MS}ms`;
        close();
      }, RUN_TIMEOUT_MS);
      return;
    }
    if (frame.method !== "graph_execution_event" || !frame.data) return;
    if (frame.data.id !== run.execId) return;
    wsEvents.add(1, tags);
    const now = Date.now();
    const status = frame.data.status;
    if (status === "RUNNING" && run.startedAt === null) {
      run.startedAt = now;
      graphRunStart.add(now - run.executedAt, tags);
      const stamp = Date.parse(frame.data.started_at);
      if (!Number.isNaN(stamp)) wsEventLatency.add(now - stamp, tags);
    }
    if (TERMINAL.includes(status)) {
      run.status = status;
      run.endedAt = now;
      const stamp = Date.parse(frame.data.ended_at);
      if (!Number.isNaN(stamp)) wsEventLatency.add(now - stamp, tags);
      close();
    }
  };
}

// The socket failed before the run reached a terminal status; the first
// reason given is kept.
function socketFailed(run, why) {
  if (run.status !== null) return;
  run.socketFailed = true;
  run.error = run.error || why;
}

function startRun(data, graph, run, phase) {
  run.executedAt = Date.now();
  const res = api("POST", `/graphs/${graph.graphId}/execute/${graph.version}`, {
    data,
    body: { inputs: graph.inputs, credentials_inputs: {}, source: "library" },
    route: "/graphs/{graph_id}/execute/{graph_version}",
    endpoint: "graphs.execute",
    phase,
    expect: [200],
  });
  if (res.ok && res.json && res.json.id) {
    run.execId = res.json.id;
  } else {
    run.error = run.error || `execute answered HTTP ${res.res.status}`;
  }
}

// After the socket has closed, or with no socket at all: a terminal status
// came over the socket, the run timed out, or the socket failed. A run the
// socket could not start is started here; polling settles what the socket
// did not; then the detail and list reads follow, as the page does them.
function finishRun(data, graph, run, phase, tags) {
  if (run.execId === null && run.socketFailed) {
    run.source = "poll";
    startRun(data, graph, run, phase);
  }
  if (run.execId !== null && run.status === null) {
    run.source = "poll";
    pollUntilTerminal(data, graph, run, phase);
  }
  if (run.status !== null) {
    api("GET", `/graphs/${graph.graphId}/executions/${run.execId}`, {
      data,
      route: "/graphs/{graph_id}/executions/{graph_exec_id}",
      endpoint: "graphs.execution_detail",
      phase,
    });
    api("GET", `/graphs/${graph.graphId}/executions?page=1&page_size=25`, {
      data,
      route: "/graphs/{graph_id}/executions",
      endpoint: "graphs.executions_list",
      phase,
    });
  }
  record(run, tags);
}

// Polling gets the rest of the run's timeout, and never less than
// POLL_GRACE_MS from when it starts: the socket timer has used up the
// timeout by the time it hands a run over.
function pollUntilTerminal(data, graph, run, phase) {
  const deadline = Math.max(
    run.executedAt + RUN_TIMEOUT_MS,
    Date.now() + POLL_GRACE_MS,
  );
  while (Date.now() < deadline) {
    sleep(1);
    const res = api(
      "GET",
      `/graphs/${graph.graphId}/executions/${run.execId}`,
      {
        data,
        route: "/graphs/{graph_id}/executions/{graph_exec_id}",
        endpoint: "graphs.execution_detail",
        phase,
      },
    );
    const status = res.json && res.json.status;
    if (status === "RUNNING" && run.startedAt === null) {
      run.startedAt = Date.now();
    }
    if (TERMINAL.includes(status)) {
      run.status = status;
      run.endedAt = Date.now();
      return;
    }
  }
  const waited = `not finished ${Math.round((Date.now() - run.executedAt) / 1000)}s after execute`;
  run.error = run.error ? `${run.error}; ${waited}` : waited;
}

function record(run, tags) {
  const sourceTags = Object.assign({ source: run.source }, tags);
  graphRuns.add(1, sourceTags);
  wsErrors.add(run.socketFailed, tags);
  const completed = run.status === "COMPLETED";
  check(
    run,
    {
      "graphs.run: started": (r) => r.execId !== null,
      "graphs.run: completed": () => completed,
    },
    tags,
  );
  if (completed) {
    graphRunTime.add(run.endedAt - run.executedAt, sourceTags);
    graphRunErrors.add(false, tags);
  } else {
    graphRunsFailed.add(1, sourceTags);
    graphRunErrors.add(true, tags);
  }
  if (!completed || run.socketFailed) {
    console.warn(
      `graph run ${run.execId || "(not started)"} ${run.status || "unfinished"} (${run.source}): ${withoutToken(
        run.error || "no error",
      )}`,
    );
  }
}
