import {
  expect,
  type Locator,
  type Page,
  type Response,
} from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { waitForAgentPageLoad } from "../pages/library.page";
import { type Agent, openRunDialog } from "./typed-inputs";

// The Home "Recent work" journey's steps: a workflow saved with the request
// the builder's Save sends, run and scheduled from its Library page the way a
// person does, then read off Home's Recent work card.

const STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9";

/** What Home calls a workflow it can't name (DEFAULT_AGENT_NAME in
 * backend/copilot/briefing/outcome.py). */
export const UNNAMED_WORKFLOW = "Workflow task";

export interface Workflow extends Agent {
  name: string;
}

export interface Schedule {
  id: string;
  name: string;
}

interface HomeDashboard {
  recent_work?: {
    groups?: { items?: { category: string; title: string }[] }[];
  } | null;
}

/** Saves a one-step workflow (Store Value, no inputs) as the builder does. */
export async function createWorkflow(
  page: Page,
  prefix: string,
): Promise<Workflow> {
  const name = createUniqueAgentName(prefix);
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "builder",
      graph: {
        name,
        description: "Stores a value, for the Home recent work journey",
        is_active: true,
        nodes: [
          {
            id: "store",
            block_id: STORE_VALUE_BLOCK_ID,
            input_default: { input: "done" },
            metadata: { position: { x: 0, y: 0 } },
            input_links: [],
            output_links: [],
          },
        ],
        links: [],
      },
    },
  });
  expect(response.ok(), `saving ${name}: ${response.status()}`).toBe(true);
  const { id: graphID } = (await response.json()) as { id: string };
  const entry = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(entry.ok(), `reading ${name}'s Library entry: ${entry.status()}`).toBe(
    true,
  );
  return {
    name,
    graphID,
    libraryAgentID: ((await entry.json()) as { id: string }).id,
  };
}

/** Schedules the workflow from its Library page's run dialog, keeping the
 * dialog's own timing (weekly, Monday 09:00). */
export async function scheduleInLibrary(
  page: Page,
  workflow: Workflow,
  name: string,
): Promise<Schedule> {
  const runDialog = await openRunDialog(page, workflow);
  await runDialog.getByRole("button", { name: "Schedule Task" }).click();
  const dialog = page.getByRole("dialog", { name: "Schedule Run" });
  await dialog.getByLabel("Schedule Name").fill(name);
  // Saving waits on the scheduler service, which is slow on a cold stack.
  const [response] = await Promise.all([
    page.waitForResponse(isScheduleCreate(workflow), { timeout: 30_000 }),
    dialog.getByRole("button", { name: "Schedule", exact: true }).click(),
  ]);
  expect(
    response.ok(),
    `scheduling ${workflow.name}: ${response.status()}`,
  ).toBe(true);
  await expect(dialog).toBeHidden({ timeout: 30_000 });
  return { id: ((await response.json()) as { id: string }).id, name };
}

/** Deletes the schedule from the workflow's Library page. */
export async function deleteScheduleInLibrary(
  page: Page,
  workflow: Workflow,
  schedule: Schedule,
) {
  await page.goto(workflowLink(workflow));
  await waitForAgentPageLoad(page);
  await page.getByRole("tab", { name: /^Scheduled/ }).click();
  await page.getByText(schedule.name, { exact: true }).first().click();
  await expect(page.getByText("Next run", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete schedule" }).first().click();
  await page.getByRole("button", { name: "Delete Schedule" }).click();
  await expect
    .poll(() => listSchedules(page, workflow), {
      message: "the Library should delete the schedule",
    })
    .toEqual([]);
}

export async function listSchedules(page: Page, workflow: Workflow) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graphID}/schedules`,
  );
  expect(response.ok(), `listing schedules: ${response.status()}`).toBe(true);
  return (await response.json()) as Schedule[];
}

/** Deletes a schedule through the API; `mayBeGone` for a cleanup that runs
 * whether or not the test deleted it first. */
export async function deleteSchedule(
  page: Page,
  scheduleID: string,
  { mayBeGone = false } = {},
) {
  const response = await page.request.delete(
    `/api/proxy/api/schedules/${scheduleID}`,
  );
  const deleted = response.ok() || (mayBeGone && response.status() === 404);
  expect(deleted, `deleting schedule ${scheduleID}: ${response.status()}`).toBe(
    true,
  );
}

/** Retires Otto's first greeting, as sending the first message does. Until
 * then Home shows the greeting in place of its cards, so a user who has never
 * chatted sees no Recent work card. */
export async function finishOttosGreeting(page: Page) {
  const response = await page.request.post(
    "/api/proxy/api/onboarding/brain-dump/intro/complete",
  );
  expect(response.ok(), `finishing Otto's greeting: ${response.status()}`).toBe(
    true,
  );
}

/** Opens Home and returns its Recent work card. The card draws only once the
 * dashboard has loaded, so what it shows comes from this visit's read. */
export async function openHome(page: Page) {
  const [dashboard] = await Promise.all([
    page.waitForResponse(isHomeRead, { timeout: 30_000 }),
    page.goto("/home"),
  ]);
  expect(dashboard.status(), "reading Home's dashboard").toBe(200);
  const card = page.getByRole("region", { name: "Recent work", exact: true });
  await expect(card, "Home should show its Recent work card").toBeVisible();
  return card;
}

/** Waits for Home's dashboard to list the schedule: the scheduler records a
 * new schedule's activity in the background, a moment after it answers. */
export async function waitForHomeToList(page: Page, schedule: Schedule) {
  await expect
    .poll(async () => listsSchedule(await readHome(page), schedule), {
      message: `Home's dashboard should list the schedule "${schedule.name}"`,
      timeout: 30_000,
      intervals: [1_000],
    })
    .toBe(true);
}

/** The card's group for the workflow: the one whose header links to its
 * Library page, whatever name the group is given. */
export function workflowGroup(card: Locator, workflow: Workflow) {
  return card.getByRole("article").filter({
    has: card.page().locator(`a[href="${workflowLink(workflow)}"]`),
  });
}

/** A run's row in a group, by the run it opens. */
export function runRow(group: Locator, runID: string) {
  return group.locator(`a[href$="activeItem=${runID}"]`);
}

/** A run's status as Home shows it: only a failed run's row says so. */
export async function statusOnHome(row: Locator) {
  const failed = await row.getByText("Failed", { exact: true }).count();
  return failed > 0 ? "failed" : "completed";
}

/** A run's status as the Library's task list shows it: "Task completed" is
 * "completed". */
export async function statusInLibrary(page: Page, runID: string) {
  const status = page
    .getByTestId(`library-run-item-${runID}`)
    .getByRole("img", { name: /^Task / });
  await expect(status, "the Library should list the run").toBeVisible();
  return ((await status.getAttribute("aria-label")) ?? "").replace(
    /^Task /,
    "",
  );
}

/** Who each group listing `title` belongs to: "the workflow" for the
 * workflow's own group, otherwise the group's name. */
export function groupsListing(
  card: Locator,
  workflow: Workflow,
  title: string,
) {
  return card
    .getByRole("article")
    .filter({ has: card.page().getByText(title, { exact: true }) })
    .evaluateAll(
      (groups, link) =>
        groups.map((group) =>
          group.querySelector(`a[href="${link}"]`)
            ? "the workflow"
            : (group.getAttribute("aria-label") ?? "a group with no name"),
        ),
      workflowLink(workflow),
    );
}

export function issue(number: number) {
  return {
    type: "issue",
    description: `https://github.com/Significant-Gravitas/AutoGPT/issues/${number}`,
  };
}

function workflowLink(workflow: Workflow) {
  return `/library/agents/${workflow.libraryAgentID}`;
}

function isScheduleCreate(workflow: Workflow) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith(
      `/api/graphs/${workflow.graphID}/schedules`,
    );
}

function isHomeRead(response: Response) {
  return (
    response.request().method() === "GET" &&
    new URL(response.url()).pathname.endsWith("/api/home")
  );
}

async function readHome(page: Page) {
  const response = await page.request.get("/api/proxy/api/home");
  expect(response.ok(), `reading Home: ${response.status()}`).toBe(true);
  return (await response.json()) as HomeDashboard;
}

function listsSchedule(home: HomeDashboard, schedule: Schedule) {
  return (home.recent_work?.groups ?? []).some((group) =>
    (group.items ?? []).some(
      (item) => item.category === "schedule" && item.title === schedule.name,
    ),
  );
}
