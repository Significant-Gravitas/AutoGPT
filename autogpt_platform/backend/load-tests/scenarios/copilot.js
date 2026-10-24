// Copilot journeys.
//
// copilotReads is what opening the copilot page does: the session list, the
// usage meter the page polls, the connections and experts it offers, and one
// session's detail. copilotTurn is a conversation turn: start a session (or
// reuse one of the account's), send a message, read the stream to the end,
// then fetch the session to see the reply was saved.
import exec from "k6/execution";

import { iterationPhase } from "../configs/profiles.js";
import { streamTurn } from "../lib/copilot.js";
import { COVERED } from "../lib/coverage.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick } from "../lib/random.js";

// Short requests with short answers, so the opt-in real-model run stays
// cheap. The dummy engine ignores the text.
export const MESSAGES = [
  "In one sentence, what can you help me automate?",
  "Give me one idea for a daily summary agent, in one sentence.",
  "Reply with the single word: ready.",
  "In one sentence, how do I schedule an agent to run every morning?",
  "Name one block that fetches a web page. One sentence.",
];

export function copilotReads(data, profile) {
  const phase = iterationPhase(profile);
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );
  const account = data.accounts[(__VU - 1) % data.accounts.length];

  const list = api("GET", "/chat/sessions?limit=50&offset=0", {
    data,
    endpoint: "copilot.sessions_list",
    phase,
  });
  api("GET", "/chat/usage", { data, endpoint: "copilot.usage", phase });
  api("GET", "/chat/connections", {
    data,
    endpoint: "copilot.connections",
    phase,
  });
  api("GET", "/experts/identities", {
    data,
    endpoint: "experts.identities",
    phase,
  });

  const listed =
    list.json && Array.isArray(list.json.sessions) ? list.json.sessions : [];
  const sessionId =
    pick(random, account.sessionIds) || (listed[0] && listed[0].id);
  if (sessionId) {
    api("GET", `/chat/sessions/${sessionId}?limit=50`, {
      data,
      route: "/chat/sessions/{session_id}",
      endpoint: "copilot.session_detail",
      phase,
    });
  }
}

export function copilotTurn(data, profile) {
  const phase = iterationPhase(profile);
  if (!data.copilot.enabled) {
    // Once per run: the scenario's first iteration, whichever VU runs it.
    if (exec.scenario.iterationInTest === 0) {
      console.warn(`copilot_turn: skipped: ${data.copilot.reason}`);
    }
    return;
  }
  const account = data.accounts[(__VU - 1) % data.accounts.length];
  // An account the seed could not create a chat session for sits the
  // journey out; the summary's coverage says how many accounts ran it.
  if (account.chat !== COVERED) return;
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );

  // Most turns start a new chat, as most do in production; the rest continue
  // one of the account's seeded sessions.
  let sessionId = null;
  if (random() < 0.3 && account.sessionIds.length > 0) {
    sessionId = pick(random, account.sessionIds);
  } else {
    const created = api("POST", "/chat/sessions", {
      data,
      body: {},
      endpoint: "copilot.session_create",
      phase,
    });
    sessionId = created.json && created.json.id;
    if (!sessionId) return;
  }

  const turn = streamTurn(sessionId, pick(random, MESSAGES), {
    data,
    phase,
    timeout: data.copilot.turnTimeout,
  });

  if (turn.outcome === "completed") {
    api("GET", `/chat/sessions/${sessionId}?limit=50`, {
      data,
      route: "/chat/sessions/{session_id}",
      endpoint: "copilot.session_detail",
      phase,
    });
  }
}
