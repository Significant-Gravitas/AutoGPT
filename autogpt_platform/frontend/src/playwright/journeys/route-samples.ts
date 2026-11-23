import { expect, type Page } from "@playwright/test";
import { integrationIconSrc } from "../../components/molecules/IntegrationLogo/helpers";
import { savedAssistantTexts } from "./copilot-chat";
import {
  type Expert,
  hireThroughAPI,
  listTemplates,
  MODEL_TURN_TIMEOUT,
  waitForSetup,
} from "./experts";
import { isLocalStack } from "./journey-user";
import { ADD_34, createAgent, newListing, submitDirectly } from "./marketplace";
import quickAnswer from "./model-scripts/quick-answer.json";
import { existsInPublic, type Sample } from "./route-crawl";
import { scriptedRun } from "./scripted-model";

// Data for the route crawl's dynamic routes: what is already on the stack
// (the Marketplace's listings), or what the journey makes for itself.

export interface RouteSample extends Sample {
  route: string;
}

/** A dynamic route the crawl couldn't fill in on this stack, and why. */
export interface Unsampled {
  route: string;
  why: string;
}

const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;
const SCRIPTED_MODEL_URL =
  process.env.PLAYWRIGHT_SCRIPTED_MODEL_URL ?? "http://localhost:4010";

interface StoreSkill {
  slug: string;
  title: string;
  required_providers: string[];
}

/**
 * An approved listing, its creator, an expert on the roster and two skills,
 * as the stack has them. CI's journeys leg loads the official listings, and
 * the roster with its skills.
 */
export async function sampleMarketplace(page: Page) {
  const [agents, experts, skills] = await Promise.all([
    getJSON<{
      agents: { slug: string; creator: string; agent_name: string }[];
    }>(page, "/api/proxy/api/store/agents?page_size=1"),
    getJSON<{ id: string; name: string }[]>(
      page,
      "/api/proxy/api/experts/templates",
    ),
    getJSON<{ skills: StoreSkill[] }>(
      page,
      "/api/proxy/api/store/skills?page_size=100",
    ),
  ]);
  const [agent] = agents.agents;
  const [expert] = experts;
  const creator = agent && encodeURIComponent(agent.creator);
  return {
    agent: agent && {
      route: "/marketplace/agent/[creator]/[slug]",
      path: `/marketplace/agent/${creator}/${encodeURIComponent(agent.slug)}`,
      shows: agent.agent_name,
    },
    creator: agent && {
      route: "/marketplace/creator/[creator]",
      path: `/marketplace/creator/${creator}`,
    },
    expert: expert && {
      route: "/marketplace/experts/[expertId]",
      path: `/marketplace/experts/${expert.id}`,
      shows: expert.name,
    },
    // A skill whose logo, if it shows one, public/ has.
    skill: skillSample(
      skills.skills.find((skill) => hasItsLogo(skill.required_providers[0])),
    ),
    // A skill whose first service is an MCP server (#17313).
    skillWithMCPService: skillSample(
      skills.skills.find((skill) =>
        skill.required_providers[0]?.startsWith("mcp_"),
      ),
    ),
  };
}

function skillSample(skill: StoreSkill | undefined): RouteSample | undefined {
  return (
    skill && {
      route: "/marketplace/skills/[slug]",
      path: `/marketplace/skills/${encodeURIComponent(skill.slug)}`,
      shows: skill.title,
    }
  );
}

function hasItsLogo(provider: string | undefined) {
  const logo = provider ? integrationIconSrc(provider) : null;
  return !logo || existsInPublic(logo);
}

/** A workflow of the user's own, in their Library. */
export async function makeWorkflow(page: Page) {
  const agent = await createAgent(page, ADD_34);
  const libraryAgent = await getJSON<{ id: string }>(
    page,
    `/api/proxy/api/library/agents/by-graph/${agent.graphID}`,
  );
  return { ...agent, libraryAgentID: libraryAgent.id };
}

/**
 * Hires the roster expert with the least to install and nothing scheduled,
 * so no schedule of its fires into later journeys, and waits for its setup.
 */
export async function hireAnExpert(page: Page): Promise<Expert | null> {
  const [template] = (await listTemplates(page))
    .filter((t) => !t.workflows.some((workflow) => workflow.schedule_cron))
    .sort((a, b) => a.workflows.length - b.workflows.length);
  if (!template) return null;
  const hired = await hireThroughAPI(page, template.id);
  return waitForSetup(page, hired.id);
}

/** Lets the expert go, as Team's Let go does. */
export async function letGo(page: Page, expert: Expert) {
  const response = await page.request.delete(
    `/api/proxy/api/experts/${expert.id}`,
  );
  return response.ok() || response.status() === 404
    ? null
    : `letting the expert go: ${response.status()}`;
}

/** Runs a workflow to the end and shares the run, as Share results does. */
export async function shareARun(page: Page) {
  const agent = await createAgent(page, ADD_34);
  const started = await page.request.post(
    `/api/proxy/api/graphs/${agent.graphID}/execute/${agent.version}`,
    { data: { inputs: { number: 8 }, credentials_inputs: {} } },
  );
  expect(started.ok(), `starting the run: ${started.status()}`).toBe(true);
  const { id: runID } = (await started.json()) as { id: string };
  const runPath = `/api/proxy/api/graphs/${agent.graphID}/executions/${runID}`;
  await expect
    .poll(
      async () => (await getJSON<{ status: string }>(page, runPath)).status,
      { message: "the run should end", timeout: 30_000 },
    )
    .toMatch(RUN_ENDED);
  expect(
    (await getJSON<{ status: string }>(page, runPath)).status,
    "the run to share should complete",
  ).toBe("COMPLETED");
  const shared = await page.request.post(`${runPath}/share`, { data: {} });
  expect(shared.ok(), `sharing the run: ${shared.status()}`).toBe(true);
  const { share_token: token } = (await shared.json()) as {
    share_token: string;
  };
  return { token, graphName: agent.name };
}

/**
 * A chat with one answer in it, shared. The answer comes from the scripted
 * model, which only a local stack started with docker-compose.journeys.yml
 * has: elsewhere this returns null.
 */
export async function shareAChat(page: Page) {
  if (!(await scriptedModelIsUp())) return null;
  const run = scriptedRun("quick-answer");
  const answer = quickAnswer.turns[0].text;
  const created = await page.request.post("/api/proxy/api/chat/sessions", {
    data: {},
  });
  expect(created.ok(), `starting a chat: ${created.status()}`).toBe(true);
  const { id: sessionID } = (await created.json()) as { id: string };
  const turn = await page.request.post(
    `/api/proxy/api/chat/sessions/${sessionID}/stream`,
    {
      data: { message: `Give me a short answer. ${run.marker}` },
      timeout: MODEL_TURN_TIMEOUT,
    },
  );
  expect(turn.ok(), `sending the message: ${turn.status()}`).toBe(true);
  await turn.body();
  await expect
    .poll(() => savedAssistantTexts(page, sessionID), {
      message: "the chat should keep the answer",
    })
    .toContain(answer);
  const shared = await page.request.post(
    `/api/proxy/api/chat/sessions/${sessionID}/share`,
    { data: { auto_share_executions: false } },
  );
  expect(shared.ok(), `sharing the chat: ${shared.status()}`).toBe(true);
  const { share_token: token } = (await shared.json()) as {
    share_token: string;
  };
  return { token, answer };
}

/** A listing the admin can preview: submitted, waiting for review. */
export async function submitAListing(page: Page) {
  const agent = await createAgent(page, ADD_34);
  const listing = newListing();
  const submission = await submitDirectly(page, agent, listing);
  return { versionID: submission.listing_version_id, title: listing.title };
}

async function scriptedModelIsUp() {
  if (!isLocalStack()) return false;
  return fetch(`${SCRIPTED_MODEL_URL}/healthz`)
    .then((response) => response.ok)
    .catch(() => false);
}

async function getJSON<T>(page: Page, path: string): Promise<T> {
  const response = await page.request.get(path);
  expect(
    response.ok(),
    `GET ${path}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as T;
}
