import { expect, type Locator, type Page } from "@playwright/test";

// The experts journey's steps. Its expert comes from the published roster
// (the skills catalog, which the stack's publish_skills service loads), and
// its scheduled workflow is an official marketplace listing, which CI's
// journeys leg loads first (platform-fullstack-ci.yml). A local stack needs
// `make load-store-agents` before `make publish-skills` for the same.

// A model turn starts a Claude Agent SDK session, which takes a while on CI.
export const MODEL_TURN_TIMEOUT = 60_000;
// A hire installs its workflows, skills and routines in the background.
const SETUP_TIMEOUT = 60_000;

const BLOCK = {
  input: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  output: "363ae599-353e-4804-937e-b2ee3cef3da4",
} as const;

export interface ExpertWorkflow {
  id: string;
  name: string | null;
  graph_id: string | null;
  library_agent_id: string | null;
  store_listing_version_id: string | null;
  schedule_cron: string | null;
  schedule_id: string | null;
}

export interface Expert {
  id: string;
  name: string;
  role: string;
  job_title: string | null;
  is_archived: boolean;
  source_template_id: string | null;
  setup_status: "installing" | "ready" | "failed";
  setup_failures: string[];
  workflows: ExpertWorkflow[];
  voice_samples: { label: string; text: string }[];
  credential_providers: string[];
  schedules_paused_at: string | null;
}

export interface SetupItem {
  expert_id: string;
  workflow_id: string;
  workflow_name: string | null;
  providers: string[];
  resolution: "connect" | "allow" | "inputs" | "workflow";
  missing_inputs: string[];
}

export interface Schedule {
  id: string;
  name: string;
  graph_id: string;
  expert_id: string | null;
  next_run_time: string | null;
}

export interface OwnWorkflow {
  name: string;
  graphID: string;
  libraryAgentID: string;
  /** The one input it needs, which has no default. */
  input: string;
}

// ── The roster ─────────────────────────────────────────────────────────

export async function listTemplates(page: Page) {
  return getJSON<Expert[]>(page, "/api/proxy/api/experts/templates");
}

/**
 * The roster expert whose hire brings a workflow on a cadence: the one that
 * can need setup on Team. It also needs a job title to be found by, and
 * writing samples, since its hire's voice pick is where the journey puts the
 * model script's marker.
 */
export function scheduledExpert(templates: Expert[]) {
  const expert = templates.find(
    (template) =>
      template.job_title &&
      template.voice_samples.length > 0 &&
      template.workflows.some((workflow) => workflow.schedule_cron),
  );
  if (!expert) {
    throw new Error(
      "no roster expert has a scheduled workflow, a job title and writing " +
        "samples. The roster's workflows are official marketplace listings: " +
        "load them before the roster is published (CI's journeys leg does; " +
        "locally, make load-store-agents, then make publish-skills). Roster: " +
        JSON.stringify(
          templates.map((t) => [t.name, t.job_title, t.workflows.length]),
        ),
    );
  }
  return expert;
}

// ── Hiring ─────────────────────────────────────────────────────────────

/**
 * Hires through the API, as the trial's team hire does: no page follows it,
 * so no kickoff turn starts. The page's own hire is the first test's.
 */
export async function hireThroughAPI(page: Page, templateID: string) {
  const response = await page.request.post("/api/proxy/api/experts", {
    data: { template_id: templateID, surface: "onboarding" },
  });
  expect(
    response.ok(),
    `hiring ${templateID}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return ((await response.json()) as { expert: Expert }).expert;
}

export async function readExpert(page: Page, expertID: string) {
  return getJSON<Expert>(page, `/api/proxy/api/experts/${expertID}`);
}

export async function listHiredExperts(page: Page) {
  return getJSON<Expert[]>(page, "/api/proxy/api/experts");
}

/** Waits for the hire's background setup, and returns the expert after it. */
export async function waitForSetup(page: Page, expertID: string) {
  await expect
    .poll(async () => (await readExpert(page, expertID)).setup_status, {
      message: "the hire should finish installing",
      timeout: SETUP_TIMEOUT,
    })
    .not.toBe("installing");
  return readExpert(page, expertID);
}

export function scheduledWorkflow(expert: Expert) {
  const workflow = expert.workflows.find((w) => w.schedule_cron);
  expect(
    workflow,
    `the hire should install the roster's scheduled workflow; it has ` +
      JSON.stringify(expert.workflows.map((w) => [w.name, w.schedule_cron])) +
      `, setup ${expert.setup_status} ${JSON.stringify(expert.setup_failures)}`,
  ).toBeTruthy();
  return workflow as ExpertWorkflow;
}

export async function listSetupItems(page: Page, expertID: string) {
  const items = await getJSON<SetupItem[]>(
    page,
    "/api/proxy/api/experts/setup",
  );
  return items.filter((item) => item.expert_id === expertID);
}

export async function grantedProviders(page: Page, expertID: string) {
  const granted = await getJSON<{ provider: string; title: string }[]>(
    page,
    `/api/proxy/api/experts/${expertID}/credentials`,
  );
  return granted.map((credential) => [credential.provider, credential.title]);
}

/** The setup still pending on the expert: what an onboarding card asks. */
export async function pendingOnboarding(page: Page, expertID: string) {
  return getJSON<{ session_id: string } | null>(
    page,
    `/api/proxy/api/experts/${expertID}/onboarding`,
  );
}

export async function listExpertChats(page: Page, expertID: string) {
  const body = await getJSON<{ sessions: { id: string }[] }>(
    page,
    `/api/proxy/api/chat/sessions?expert_id=${expertID}&limit=20&pinned_first=false`,
  );
  return body.sessions.map((session) => session.id);
}

// ── The user's own workflow and its schedules ──────────────────────────

/** Saves a workflow that needs one input and no account: topic → summary. */
export async function createOwnWorkflow(page: Page, name: string) {
  const input = "topic";
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name,
        description: "Made by the experts journey",
        is_active: true,
        nodes: [
          node("topic", BLOCK.input, { name: input, title: "Topic" }, 0),
          node(
            "summary",
            BLOCK.output,
            { name: "summary", title: "Summary" },
            600,
          ),
        ],
        links: [
          {
            source_id: "topic",
            source_name: "result",
            sink_id: "summary",
            sink_name: "value",
          },
        ],
      },
    },
  });
  expect(response.ok(), `saving ${name}: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string };
  const agent = await getJSON<{ id: string }>(
    page,
    `/api/proxy/api/library/agents/by-graph/${graph.id}`,
  );
  return { name, graphID: graph.id, libraryAgentID: agent.id, input };
}

export async function installOnExpert(
  page: Page,
  expertID: string,
  workflow: OwnWorkflow,
) {
  const response = await page.request.post(
    `/api/proxy/api/experts/${expertID}/workflows`,
    { data: { library_agent_id: workflow.libraryAgentID } },
  );
  expect(
    response.ok(),
    `installing ${workflow.name}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as ExpertWorkflow;
}

/** Schedules the workflow the way its run dialog does, naming no expert. */
export async function scheduleThroughAPI(
  page: Page,
  workflow: OwnWorkflow,
  name: string,
) {
  const response = await page.request.post(
    `/api/proxy/api/graphs/${workflow.graphID}/schedules`,
    {
      data: {
        name,
        cron: LEAP_DAY,
        inputs: { [workflow.input]: `Scheduled ${name}` },
        credentials: {},
        timezone: "UTC",
      },
    },
  );
  expect(
    response.ok(),
    `scheduling ${name}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as Schedule;
}

// Midnight on 29 February: a real cadence that won't fire while a journey
// runs (the next one is in 2028).
export const LEAP_DAY = "0 0 29 2 *";

export async function listSchedules(page: Page, graphID: string) {
  return getJSON<Schedule[]>(
    page,
    `/api/proxy/api/graphs/${graphID}/schedules`,
  );
}

/** Each schedule's name, and whether it will fire, by name. */
export async function scheduleStates(page: Page, graphID: string) {
  return (await listSchedules(page, graphID))
    .map((schedule): [string, string] => [
      schedule.name,
      schedule.next_run_time ? "active" : "paused",
    ])
    .sort(([a], [b]) => a.localeCompare(b));
}

/** A schedule left behind fires into the next journey's stack. Pass the ids
 *  the test made as well: an archived expert's schedules drop out of the
 *  list, so a test that failed before re-hiring would leave them. */
export async function deleteSchedules(
  page: Page,
  graphID: string,
  madeIDs: string[] = [],
) {
  // Retried a little, then reported rather than read as "none left": the
  // ids made still go, but a schedule the list would have shown would stay.
  // Soft, so a report from a `finally` can't hide the test's own failure.
  let listed: Schedule[] = [];
  await expect
    .configure({ soft: true })
    .poll(
      async () => {
        try {
          listed = await listSchedules(page, graphID);
          return "listed";
        } catch (error) {
          return `${error}`;
        }
      },
      {
        message: `listing graph ${graphID}'s schedules to delete them`,
        timeout: 10_000,
      },
    )
    .toBe("listed");
  const ids = new Set([...madeIDs, ...listed.map((schedule) => schedule.id)]);
  for (const id of ids) {
    const response = await page.request.delete(
      `/api/proxy/api/schedules/${id}`,
    );
    expect
      .soft(
        response.ok() || response.status() === 404,
        `deleting schedule ${id}: ${response.status()}`,
      )
      .toBe(true);
  }
}

// ── The page ───────────────────────────────────────────────────────────

/** An expert's card on Team. Its name link is the card's own child. */
export function teamCard(page: Page, expertName: string) {
  return page
    .getByRole("link", { name: `View ${expertName}`, exact: true })
    .locator("xpath=..");
}

/** The hire's onboarding card, by its progress bar. */
export function onboardingCard(scope: Locator) {
  return scope
    .getByRole("progressbar", { name: "Setup progress" })
    .locator("xpath=..");
}

export async function expectQuestion(
  card: Locator,
  progress: string,
  question: string,
) {
  await expect(
    card.getByRole("progressbar", { name: "Setup progress" }),
    `the card should be at "${progress}"`,
  ).toHaveAttribute("aria-valuetext", progress);
  await expect(card.getByRole("radiogroup", { name: question })).toBeVisible();
}

/** Picks an option. A pick moves on by itself, after a short beat. */
export async function pickOption(card: Locator, option: string) {
  await card.getByRole("radio", { name: option, exact: true }).click();
}

export function isPath(method: string, path: string | RegExp) {
  return (response: { request(): { method(): string }; url(): string }) =>
    response.request().method() === method &&
    (typeof path === "string"
      ? new URL(response.url()).pathname === path
      : path.test(new URL(response.url()).pathname));
}

async function getJSON<T>(page: Page, path: string): Promise<T> {
  const response = await page.request.get(path);
  expect(
    response.ok(),
    `GET ${path}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as T;
}

function node(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  x: number,
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x, y: 0 } },
    input_links: [],
    output_links: [],
  };
}
