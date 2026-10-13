// A graph run the way the library page watches one: open the WebSocket,
// subscribe to the graph's executions, start a run, follow its events to a
// terminal status, then confirm with the execution's detail and list the
// graph's executions.
//
// Measured: ws_connect_ms (the socket opens), graph_run_start_ms (execute
// to the first RUNNING event), graph_run_ms (execute to COMPLETED),
// ws_event_latency_ms (an event's own ended_at to its arrival, so it
// includes any clock skew between the target and this machine). A run that
// gets no terminal event within the timeout is checked by polling and
// counted as failed if it did not complete.
import exec from "k6/execution";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { setTimeout, clearTimeout } from "k6/timers";
import { WebSocket } from "k6/websockets";

import { getEnvironmentConfig } from "../configs/environment.js";
import { currentPhase } from "../configs/profiles.js";
import { authHeaders } from "../lib/auth.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick } from "../lib/random.js";

const config = getEnvironmentConfig();

export const graphRunStart = new Trend("graph_run_start_ms", true);
export const graphRunTime = new Trend("graph_run_ms", true);
export const wsConnect = new Trend("ws_connect_ms", true);
export const wsEventLatency = new Trend("ws_event_latency_ms", true);
export const wsEvents = new Counter("ws_events");
export const graphRuns = new Counter("graph_runs");
export const graphRunsFailed = new Counter("graph_runs_failed");
export const graphRunErrors = new Rate("graph_run_errors");

const TERMINAL = ["COMPLETED", "FAILED", "TERMINATED"];
const RUN_TIMEOUT_MS = Number(__ENV.GRAPH_RUN_TIMEOUT_MS || 60000);
const WS_OPEN_TIMEOUT_MS = 10000;

let announcedSkip = false;

export function graphRun(data, profile) {
  if (!data.graphs.enabled) {
    if (!announcedSkip) {
      console.warn(`graph_run: skipped: ${data.graphs.reason}`);
      announcedSkip = true;
    }
    return;
  }
  const phase = currentPhase(profile);
  const random = iterationRandom(exec.scenario.name, exec.scenario.iterationInTest);
  const account = data.accounts[(__VU - 1) % data.accounts.length];
  const graph = pick(random, account.graphs);
  if (!graph) return;

  const tags = { phase, endpoint: "graphs.run" };
  const run = {
    execId: null,
    executedAt: null,
    startedAt: null,
    endedAt: null,
    status: null,
    source: "ws",
    error: null,
  };

  const token = authHeaders(data).Authorization.replace(/^Bearer /, "");
  const connectAt = Date.now();
  let socket;
  try {
    socket = new WebSocket(`${config.WS_URL}?token=${encodeURIComponent(token)}`);
  } catch (e) {
    run.error = `websocket: ${e}`;
  }

  if (!socket) {
    finishByPolling(data, graph, run, phase, tags);
    return;
  }

  let timer = setTimeout(() => {
    run.error = run.error || "no open event from the websocket";
    close();
  }, WS_OPEN_TIMEOUT_MS);

  function close() {
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
    wsConnect.add(Date.now() - connectAt, tags);
    socket.send(
      JSON.stringify({
        method: "subscribe_graph_executions",
        data: { graph_id: graph.graphId },
      }),
    );
  };

  socket.onerror = (e) => {
    run.error = run.error || `websocket error: ${e && e.error ? e.error : e}`;
    close();
  };

  // The iteration ends when the socket has closed and no timer is pending,
  // so the reads that follow the run happen from here.
  socket.onclose = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    finishRun(data, graph, run, phase, tags);
  };

  socket.onmessage = (message) => {
    let frame;
    try {
      frame = JSON.parse(message.data);
    } catch (e) {
      return;
    }
    if (frame.method === "subscribe_graph_executions" && run.execId === null) {
      if (frame.success === false) {
        run.error = `subscribe refused: ${frame.error}`;
        close();
        return;
      }
      // Subscribed before the run exists, so no event is missed.
      startRun(data, graph, run, phase);
      if (run.execId === null) {
        close();
        return;
      }
      clearTimeout(timer);
      timer = setTimeout(() => {
        run.error = run.error || `no terminal event within ${RUN_TIMEOUT_MS}ms`;
        close();
      }, RUN_TIMEOUT_MS);
      return;
    }
    if (frame.method !== "graph_execution_event" || !frame.data) return;
    if (run.execId === null || frame.data.id !== run.execId) return;
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
    run.error = `execute answered HTTP ${res.res.status}`;
  }
}

// After the socket closes: a terminal status came over the socket, or the
// run timed out, or the socket failed. Polling settles what the socket did
// not, then the detail and list reads follow, as the page does them.
function finishRun(data, graph, run, phase, tags) {
  if (run.execId === null) {
    if (run.error && /websocket|open event/.test(run.error)) {
      run.source = "poll";
      run.error = null;
      finishByPolling(data, graph, run, phase, tags);
      return;
    }
    record(run, tags);
    return;
  }
  if (run.status === null) {
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

function finishByPolling(data, graph, run, phase, tags) {
  startRun(data, graph, run, phase);
  if (run.execId !== null) pollUntilTerminal(data, graph, run, phase);
  record(run, tags);
}

function pollUntilTerminal(data, graph, run, phase) {
  const deadline = run.executedAt + RUN_TIMEOUT_MS;
  while (Date.now() < deadline) {
    sleep(1);
    const res = api("GET", `/graphs/${graph.graphId}/executions/${run.execId}`, {
      data,
      route: "/graphs/{graph_id}/executions/{graph_exec_id}",
      endpoint: "graphs.execution_detail",
      phase,
    });
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
  run.error = run.error || `not finished within ${RUN_TIMEOUT_MS}ms`;
}

function record(run, tags) {
  const sourceTags = Object.assign({ source: run.source }, tags);
  graphRuns.add(1, sourceTags);
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
    console.warn(
      `graph run ${run.execId || "(not started)"} ${run.status || "unfinished"}: ${run.error || "no error"}`,
    );
  }
}
