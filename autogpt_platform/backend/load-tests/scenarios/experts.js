// Expert journeys, both as the dummy accounts, whose turns run on the dummy
// engine.
//
// expertWork is working with a hired expert, which users do far more often
// than they hire one: the expert's page (the team list, the expert, its
// activity graph, the schedules, home, its recent chats and the files it
// made), then one of its chats, a seeded one or a new one, and a turn in it.
// expertHire is hiring one from the marketplace: the templates, the hire,
// the composer's reads, the kickoff session and its first turn, then the
// team page polling while the hire installs; it cleans up after itself, so
// the account stays under its cap of 20 active experts.
//
// Each takes its accounts in turn, by iteration, as the turn journeys do
// (scenarios/copilot.js).
import { sleep } from "k6";
import exec from "k6/execution";

import { iterationPhase } from "../configs/profiles.js";
import { KICKOFF_PROMPT, streamTurn } from "../lib/copilot.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick, randomUuid } from "../lib/random.js";
import {
  MESSAGES,
  iterationAccount,
  sessionDetail,
  sessionList,
  skipOnce,
} from "./copilot.js";

// The team page polls the list every 2 s while an expert is installing
// (frontend: team/useTeamPage.ts); the journey stops after 15 polls.
const SETUP_POLLS = 15;
const SETUP_POLL_SECONDS = 2;

// Per VU, the accounts it has found with every template they may hire hired,
// so each is logged once.
const exhausted = {};

function listExperts(data, account, phase) {
  const listed = api("GET", "/experts", {
    data,
    account,
    endpoint: "experts.list",
    phase,
  });
  return Array.isArray(listed.json) ? listed.json : [];
}

function latestExpertChat(data, account, expertId, phase) {
  return api(
    "GET",
    `/chat/sessions?expert_id=${expertId}&limit=1&pinned_first=false`,
    {
      data,
      account,
      route: "/chat/sessions",
      endpoint: "experts.sessions",
      phase,
    },
  );
}

export function expertWork(data, profile) {
  const phase = iterationPhase(profile);
  const work = data.experts.work;
  if (!work.enabled) {
    skipOnce("expert_work", work.reason);
    return;
  }
  const iteration = exec.scenario.iterationInTest;
  const random = iterationRandom(exec.scenario.name, iteration);
  const account = iterationAccount(data, work.accounts, iteration);
  const expertId = account.experts.workId;

  // The expert's page.
  listExperts(data, account, phase);
  api("GET", `/experts/${expertId}`, {
    data,
    account,
    route: "/experts/{expert_id}",
    endpoint: "experts.detail",
    phase,
  });
  api("GET", `/experts/${expertId}/activity`, {
    data,
    account,
    route: "/experts/{expert_id}/activity",
    endpoint: "experts.activity",
    phase,
  });
  api("GET", "/schedules", {
    data,
    account,
    endpoint: "schedules.list",
    phase,
  });
  api("GET", "/home", { data, account, endpoint: "home", phase });
  api("GET", `/chat/sessions?expert_id=${expertId}&limit=5`, {
    data,
    account,
    route: "/chat/sessions",
    endpoint: "experts.sessions",
    phase,
  });
  api(
    "GET",
    `/workspace/files?expert_id=${expertId}&origin=generated&limit=200`,
    {
      data,
      account,
      route: "/workspace/files",
      endpoint: "experts.files",
      phase,
    },
  );
  latestExpertChat(data, account, expertId, phase);

  // Most of the time the user goes back to one of the expert's chats; the
  // rest of the time they start a new one, named by the page.
  let sessionId = null;
  if (random() < 0.7 && account.experts.sessionIds.length > 0) {
    sessionId = pick(random, account.experts.sessionIds);
    sessionDetail(data, account, sessionId, phase);
  } else {
    const created = api("POST", "/chat/sessions", {
      data,
      account,
      body: { expert_id: expertId, session_id: randomUuid() },
      endpoint: "experts.session_create",
      phase,
    });
    sessionId = created.json && created.json.id;
  }
  // The turn needs the dummy engine; without it the journey is the page.
  if (!sessionId || !data.copilot.dummy.enabled) return;

  const turn = streamTurn(sessionId, pick(random, MESSAGES), {
    data,
    account,
    phase,
    engine: "dummy",
    timeout: data.copilot.timeouts.dummy,
  });
  if (!turn.finish) return;
  sessionDetail(data, account, sessionId, phase);
  sessionList(data, account, phase);
}

// The template an iteration hires: the account's own list of templates it
// may hire, one after the other, so a run never hires the same template on
// an account twice. A second hire of an archived template revives the
// archived expert, a different path from a hire, so an account that has
// hired them all hires no more.
function hireTemplate(data, iteration) {
  const hire = data.experts.hire;
  const account = iterationAccount(data, hire.accounts, iteration);
  const round = Math.floor(iteration / hire.accounts.length);
  return { account, templateId: account.experts.hireTemplateIds[round] };
}

export function expertHire(data, profile) {
  const phase = iterationPhase(profile);
  const hire = data.experts.hire;
  if (!hire.enabled) {
    skipOnce("expert_hire", hire.reason);
    return;
  }
  const iteration = exec.scenario.iterationInTest;
  const { account, templateId } = hireTemplate(data, iteration);
  if (!templateId) {
    if (!exhausted[account.index]) {
      exhausted[account.index] = true;
      console.warn(
        `expert_hire: account ${account.index + 1} has hired every template it may (${account.experts.hireTemplateIds.length}); its later iterations hire nothing`,
      );
    }
    return;
  }

  // The marketplace's expert page, signed in, and the hire.
  api("GET", "/experts/templates", {
    data,
    account,
    endpoint: "experts.templates",
    phase,
  });
  listExperts(data, account, phase);
  api("GET", "/integrations/providers/system", {
    data,
    account,
    endpoint: "account.providers_system",
    phase,
  });
  const hired = api("POST", "/experts", {
    data,
    account,
    body: { template_id: templateId, surface: "expert_page" },
    endpoint: "experts.hire",
    phase,
  });
  const expert = hired.json && hired.json.expert;
  if (!expert || !expert.id) return;
  const expertId = expert.id;
  // A supplied account that had this expert keeps it.
  const ours = account.experts.preexisting.indexOf(expertId) < 0;

  // The new expert's chat opens: the composer's reads, then the kickoff
  // session, which the server picks the id of, and its first turn.
  let status = statusOf(listExperts(data, account, phase), expertId);
  api("GET", "/experts/identities", {
    data,
    account,
    endpoint: "experts.identities",
    phase,
  });
  latestExpertChat(data, account, expertId, phase);
  api("GET", "/chat/transports", {
    data,
    account,
    endpoint: "copilot.transports",
    phase,
  });
  const created = api("POST", "/chat/sessions", {
    data,
    account,
    body: { expert_id: expertId, expert_kickoff: true },
    endpoint: "experts.kickoff_create",
    phase,
  });
  const sessionId = created.json && created.json.id;
  if (sessionId) {
    if (data.copilot.dummy.enabled) {
      streamTurn(sessionId, KICKOFF_PROMPT, {
        data,
        account,
        phase,
        engine: "dummy",
        expertKickoff: true,
        timeout: data.copilot.timeouts.dummy,
      });
    }
    sessionDetail(data, account, sessionId, phase);
  }

  // The team page, while the hire's workflows, skills and routines install.
  for (let poll = 0; poll < SETUP_POLLS && status === "installing"; poll++) {
    if (poll > 0) sleep(SETUP_POLL_SECONDS);
    status = statusOf(listExperts(data, account, phase), expertId);
  }

  // Cleaning up, timed apart from the reads: the kickoff chat, then the
  // expert, archived, which pauses the schedules its setup created. An
  // expert still installing is left to teardown, which archives every
  // expert the run hired, so a schedule its setup creates after this point
  // is paused too.
  if (sessionId) {
    api("DELETE", `/chat/sessions/${sessionId}`, {
      data,
      account,
      route: "/chat/sessions/{session_id}",
      endpoint: "copilot.session_delete",
      phase,
      expect: [200, 204],
    });
  }
  if (ours && status === "installing") {
    console.warn(
      `expert_hire: expert ${expertId} was still installing after ${SETUP_POLLS} polls; teardown archives it`,
    );
  } else if (ours) {
    api("DELETE", `/experts/${expertId}`, {
      data,
      account,
      route: "/experts/{expert_id}",
      endpoint: "experts.archive",
      phase,
      expect: [200, 204],
    });
  }
}

function statusOf(experts, expertId) {
  const expert = experts.find((e) => e.id === expertId);
  return expert ? expert.setup_status : null;
}
