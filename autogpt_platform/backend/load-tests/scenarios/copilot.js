// Copilot journeys.
//
// copilotReads is what opening a chat does: the session list, the usage
// meter, the transports, connections and experts the composer offers, then
// one session's detail, its review cards and its queued messages.
// copilotPoll is a chat left open while it waits on a review card, as the
// page polls it. The turn journeys send a message and read the stream to
// the end, each as the accounts of one engine (lib/auth.js):
//
// - copilotTurn: the dummy engine, the platform's own path without a model;
//   a new session or one of the account's seeded ones, then the reads the
//   page makes after a turn;
// - copilotTurnScripted: the whole SDK engine against the scripted model,
//   always on a new session, plain or with the account's expert;
// - copilotTurnReal: the real model, a capped number of turns.
//
// A turn journey takes its accounts in turn, by iteration, so concurrent
// turns spread over them and stay under copilot's per-user caps (5 running,
// 15 in flight).
import { sleep } from "k6";
import exec from "k6/execution";

import { durationMs, iterationPhase } from "../configs/profiles.js";
import {
  recordUnstartedTurn,
  scriptedMessage,
  streamTurn,
} from "../lib/copilot.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick, randomUuid } from "../lib/random.js";

// Short requests with short answers, so the real-model turns stay cheap.
// The dummy engine ignores the text, and the scripted model answers the
// script whatever follows its marker.
export const MESSAGES = [
  "In one sentence, what can you help me automate?",
  "Give me one idea for a daily summary agent, in one sentence.",
  "Reply with the single word: ready.",
  "In one sentence, how do I schedule an agent to run every morning?",
  "Name one block that fetches a web page. One sentence.",
];

// The open chat's polling (frontend: useCopilotPendingReviews, useSessionList,
// useCopilotUsage): review cards every 2 s, the session list every 10 s and
// usage every 30 s, so ten seconds of it is five card reads, one list read
// and, every third time, a usage read.
const REVIEW_POLLS = 5;
const REVIEW_POLL_SECONDS = 2;
const USAGE_EVERY = 3;
// After a new chat's first turn the page reads the session list again for
// the title the backend writes after the reply (useSessionTitlePoll).
const TITLE_POLLS = 2;
const TITLE_POLL_SECONDS = 2;

// The account an iteration of a turn journey runs as: the journey's
// accounts (indices into data.accounts) in turn.
export function iterationAccount(data, indices, iteration) {
  return data.accounts[indices[iteration % indices.length]];
}

// Once per run: the scenario's first iteration, whichever VU runs it.
export function skipOnce(name, reason) {
  if (exec.scenario.iterationInTest === 0) {
    console.warn(`${name}: skipped: ${reason}`);
  }
}

export function sessionDetail(data, account, sessionId, phase) {
  return api("GET", `/chat/sessions/${sessionId}?limit=50`, {
    data,
    account,
    route: "/chat/sessions/{session_id}",
    endpoint: "copilot.session_detail",
    phase,
  });
}

export function sessionList(data, account, phase) {
  return api("GET", "/chat/sessions?limit=50&offset=0", {
    data,
    account,
    endpoint: "copilot.sessions_list",
    phase,
  });
}

function reviewCards(data, account, sessionId, phase) {
  return api("GET", `/review/session/${sessionId}`, {
    data,
    account,
    route: "/review/session/{chat_session_id}",
    endpoint: "copilot.review_pending",
    phase,
  });
}

function usage(data, account, phase) {
  return api("GET", "/chat/usage", {
    data,
    account,
    endpoint: "copilot.usage",
    phase,
  });
}

function pendingMessages(data, account, sessionId, phase) {
  return api("GET", `/chat/sessions/${sessionId}/messages/pending`, {
    data,
    account,
    route: "/chat/sessions/{session_id}/messages/pending",
    endpoint: "copilot.pending_peek",
    phase,
  });
}

function createSession(data, account, phase, body, endpoint) {
  const created = api("POST", "/chat/sessions", {
    data,
    account,
    body,
    endpoint,
    phase,
  });
  return created.json && created.json.id ? created.json.id : null;
}

export function copilotReads(data, profile) {
  const phase = iterationPhase(profile);
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );
  const account = data.accounts[(__VU - 1) % data.accounts.length];

  const list = sessionList(data, account, phase);
  usage(data, account, phase);
  api("GET", "/chat/transports", {
    data,
    account,
    endpoint: "copilot.transports",
    phase,
  });
  api("GET", "/chat/connections", {
    data,
    account,
    endpoint: "copilot.connections",
    phase,
  });
  api("GET", "/experts/identities", {
    data,
    account,
    endpoint: "experts.identities",
    phase,
  });

  const listed =
    list.json && Array.isArray(list.json.sessions) ? list.json.sessions : [];
  const sessionId =
    pick(random, account.sessionIds) || (listed[0] && listed[0].id);
  if (!sessionId) return;
  sessionDetail(data, account, sessionId, phase);
  reviewCards(data, account, sessionId, phase);
  api("GET", `/review/session/${sessionId}/answered`, {
    data,
    account,
    route: "/review/session/{chat_session_id}/answered",
    endpoint: "copilot.review_answered",
    phase,
  });
  pendingMessages(data, account, sessionId, phase);
}

export function copilotPoll(data, profile) {
  const phase = iterationPhase(profile);
  const iteration = exec.scenario.iterationInTest;
  const random = iterationRandom(exec.scenario.name, iteration);
  const account = data.accounts[(__VU - 1) % data.accounts.length];

  // An account the seed could not create a session for has no open chat to
  // poll the cards of; it still polls the list.
  const sessionId = pick(random, account.sessionIds);
  if (sessionId) {
    for (let poll = 0; poll < REVIEW_POLLS; poll++) {
      if (poll > 0) sleep(REVIEW_POLL_SECONDS);
      reviewCards(data, account, sessionId, phase);
    }
  }
  sessionList(data, account, phase);
  if (iteration % USAGE_EVERY === 0) usage(data, account, phase);
}

export function copilotTurn(data, profile) {
  const phase = iterationPhase(profile);
  const engine = data.copilot.dummy;
  if (!engine.enabled) {
    skipOnce("copilot_turn", engine.reason);
    return;
  }
  const iteration = exec.scenario.iterationInTest;
  const random = iterationRandom(exec.scenario.name, iteration);
  const account = iterationAccount(data, engine.accounts, iteration);

  // Most turns start a new chat, as most do in production; the rest continue
  // one of the account's seeded sessions.
  let sessionId = null;
  let created = false;
  if (random() < 0.3 && account.sessionIds.length > 0) {
    sessionId = pick(random, account.sessionIds);
  } else {
    sessionId = createSession(
      data,
      account,
      phase,
      {},
      "copilot.session_create",
    );
    if (!sessionId) {
      recordUnstartedTurn("dummy", phase);
      return;
    }
    created = true;
  }

  const turn = streamTurn(sessionId, pick(random, MESSAGES), {
    data,
    account,
    phase,
    engine: "dummy",
    timeout: data.copilot.timeouts.dummy,
  });
  if (!turn.finish) return;

  // What the page reads once the turn finishes: the saved reply, the usage
  // meter, anything queued behind it, and for a new chat its title.
  sessionDetail(data, account, sessionId, phase);
  usage(data, account, phase);
  pendingMessages(data, account, sessionId, phase);
  if (created) {
    for (let poll = 0; poll < TITLE_POLLS; poll++) {
      if (poll > 0) sleep(TITLE_POLL_SECONDS);
      const list = sessionList(data, account, phase);
      const sessions =
        list.json && Array.isArray(list.json.sessions)
          ? list.json.sessions
          : [];
      if (sessions.some((s) => s.id === sessionId && s.title)) break;
    }
  }
}

// Every scripted turn is a new session: the script's first turn answers a
// conversation with no replies yet, and a second message in the same
// session would run past the script's end. Half of them are with the
// account's expert, as a chat on the expert's page is.
export function copilotTurnScripted(data, profile) {
  const phase = iterationPhase(profile);
  const engine = data.copilot.scripted;
  if (!engine.enabled) {
    skipOnce("copilot_turn_scripted", engine.reason);
    return;
  }
  const iteration = exec.scenario.iterationInTest;
  const random = iterationRandom(exec.scenario.name, iteration);
  const account = iterationAccount(data, engine.accounts, iteration);

  const expertId =
    account.experts && account.experts.workId && random() < 0.5
      ? account.experts.workId
      : null;
  const sessionId = expertId
    ? createSession(
        data,
        account,
        phase,
        { expert_id: expertId, session_id: randomUuid() },
        "experts.session_create",
      )
    : createSession(data, account, phase, {}, "copilot.session_create");
  if (!sessionId) {
    recordUnstartedTurn("scripted", phase);
    return;
  }

  const turn = streamTurn(
    sessionId,
    scriptedMessage(`${data.runTag}-${iteration}`, pick(random, MESSAGES)),
    {
      data,
      account,
      phase,
      engine: "scripted",
      timeout: data.copilot.timeouts.scripted,
    },
  );
  if (turn.finish) sessionDetail(data, account, sessionId, phase);
}

// LOAD_TEST_COPILOT_REAL_TURNS turns, one at a time through the steady
// phase, each on a new session with a one-sentence request. Each costs
// money; the run prints the estimate before it starts and reports it.
export function copilotTurnReal(data, profile) {
  // The scenario starts when the warm-up ends.
  const phase = iterationPhase(profile, durationMs(profile.warmup));
  const engine = data.copilot.real;
  if (!engine.enabled) {
    skipOnce("copilot_turn_real", engine.reason);
    return;
  }
  const iteration = exec.scenario.iterationInTest;
  const random = iterationRandom(exec.scenario.name, iteration);
  const account = iterationAccount(data, engine.accounts, iteration);

  const sessionId = createSession(
    data,
    account,
    phase,
    {},
    "copilot.session_create",
  );
  if (!sessionId) {
    recordUnstartedTurn("real", phase);
    return;
  }
  const turn = streamTurn(sessionId, pick(random, MESSAGES), {
    data,
    account,
    phase,
    engine: "real",
    timeout: data.copilot.timeouts.real,
  });
  console.log(
    `copilot_turn_real: turn ${iteration + 1} ${turn.outcome}: first chunk ${turn.firstChunkMs}ms, first token ${turn.ttftMs}ms, ` +
      `done ${turn.turnMs}ms, ${turn.chunks} frames, ${turn.text.length} chars`,
  );
}
