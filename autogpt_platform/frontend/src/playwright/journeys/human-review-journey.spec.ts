import type { Locator, Page, Response } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import {
  assertRunOutputValue,
  getActiveItemId,
  getRunStatus as readStatusBadge,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { expect, test } from "./fixtures";

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const HUMAN_IN_THE_LOOP_BLOCK_ID = "8b2a7b3c-6e9d-4a5f-8c1b-2e3f4a5b6c7d";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const REVIEW_NAME = "Note to check";
// A run rests once it waits for a review or has ended.
const RESTING = /^(REVIEW|COMPLETED|FAILED|TERMINATED)$/;
const ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

interface Agent {
  graphID: string;
  graphVersion: number;
  libraryAgentID: string;
}

interface Run {
  status: string;
  outputs: Record<string, unknown[]>;
}

interface PendingReview {
  node_exec_id: string;
  graph_exec_id: string;
  payload: unknown;
  instructions: string;
  editable: boolean;
  status: string;
}

interface LibraryAgent {
  id: string;
  has_human_in_the_loop: boolean;
  settings: { human_in_the_loop_safe_mode?: boolean };
}

test.describe("human review journey", () => {
  test("an approved review carries the reviewer's edit to the end of the run", async ({
    page,
    journeyUser: _user,
  }) => {
    const note = `Journey note ${Date.now()}`;
    const edited = `${note} as the reviewer edited it`;
    const agent = await createReviewAgent(page);

    const runID = await startRunInLibrary(page, agent, note);
    const review = await waitForReview(page, agent, runID, note);

    // The run view the run was started from notices the pause by itself.
    await expect(
      page.getByRole("button", { name: "Reviews (1)" }),
      "the run view should offer the review once the run pauses",
    ).toBeVisible();
    const draft = reviewDraft(page);
    await expect(draft, "the review should show the note").toHaveValue(note);
    await draft.fill(edited);
    const decision = await decide(page, "Approve");
    expect(decision.sent, "the approval should carry the edit").toMatchObject({
      reviews: [
        {
          node_exec_id: review.node_exec_id,
          approved: true,
          reviewed_data: edited,
        },
      ],
    });
    expect(decision.answer).toMatchObject({
      approved_count: 1,
      rejected_count: 0,
    });

    const run = await waitForRun(
      page,
      agent,
      runID,
      ENDED,
      "the approval should resume the run to its end",
    );
    expect(run.status).toBe("COMPLETED");
    expect(
      run.outputs,
      "only the approved branch should run, with the edit",
    ).toEqual({ approved: [edited] });

    await expect
      .poll(() => readStatusBadge(page), {
        message: "the run view should show the run completed",
      })
      .toBe("completed");
    await assertRunOutputValue(page, "approved", edited);
    await expect(
      outputLabel(page, "rejected"),
      "the run view should show no rejected output",
    ).toHaveCount(0);
  });

  test("a rejected review sends the original note down the rejected branch only", async ({
    page,
    journeyUser: _user,
  }) => {
    const note = `Journey note ${Date.now()}`;
    const agent = await createReviewAgent(page);
    const runID = await startRun(page, agent, note);
    const review = await waitForReview(page, agent, runID, note);

    // A person coming back to a paused run opens it from the Library.
    await page.goto(
      `/library/agents/${agent.libraryAgentID}?activeTab=runs&activeItem=${runID}`,
    );
    await waitForAgentPageLoad(page);
    await expect(
      reviewDraft(page),
      "the paused run should show the note for review",
    ).toHaveValue(note);
    const decision = await decide(page, "Reject");
    expect(decision.sent).toMatchObject({
      reviews: [{ node_exec_id: review.node_exec_id, approved: false }],
    });
    expect(decision.answer).toMatchObject({
      approved_count: 0,
      rejected_count: 1,
    });

    const run = await waitForRun(
      page,
      agent,
      runID,
      ENDED,
      "the rejection should resume the run to its end",
    );
    expect(run.status).toBe("COMPLETED");
    expect(
      run.outputs,
      "only the rejected branch should run, with the original note",
    ).toEqual({ rejected: [note] });

    await expect
      .poll(() => readStatusBadge(page), {
        message: "the run view should show the run completed",
      })
      .toBe("completed");
    await assertRunOutputValue(page, "rejected", note);
    await expect(
      outputLabel(page, "approved"),
      "the run view should show no approved output",
    ).toHaveCount(0);
  });

  test("with its safe mode switched off in the Library, a run passes the note through unreviewed", async ({
    page,
    journeyUser: _user,
  }) => {
    const note = `Journey note ${Date.now()}`;
    const agent = await createReviewAgent(page);

    await openNewAgent(page, agent);
    await page.getByRole("button", { name: "Workflow Settings" }).click();
    const safeMode = safeModeSwitch(
      page.getByRole("dialog", { name: "Workflow Settings" }),
    );
    await expect(safeMode, "safe mode should start on").toBeChecked();
    const saved = page.waitForResponse(isSettingsSave(agent.graphID));
    await safeMode.click();
    const response = await saved;
    expect(response.ok(), `saving the setting: ${response.status()}`).toBe(
      true,
    );
    await expect(
      safeMode,
      "the switch should show safe mode off",
    ).not.toBeChecked();
    expect(
      (await readLibraryAgent(page, agent.graphID)).settings,
      "the Library should keep safe mode off",
    ).toMatchObject({ human_in_the_loop_safe_mode: false });

    const runID = await startRun(page, agent, note);
    const run = await waitForRun(
      page,
      agent,
      runID,
      RESTING,
      "the run should end",
    );
    expect(
      run.status,
      "with safe mode off the run should not stop for a review",
    ).toBe("COMPLETED");
    expect(
      run.outputs,
      "the note should come out of the approved branch unchanged",
    ).toEqual({ approved: [note] });
  });
});

// AgentInput "note" -> HumanInTheLoop, whose approved and rejected data each
// go to their own AgentOutput.
async function createReviewAgent(page: Page): Promise<Agent> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name: createUniqueAgentName("Journey human review"),
        description: "Pauses a note for review, for the human review journey",
        is_active: true,
        nodes: [
          node("note", AGENT_INPUT_BLOCK_ID, { name: "note" }, [0, 0]),
          node(
            "review",
            HUMAN_IN_THE_LOOP_BLOCK_ID,
            { name: REVIEW_NAME, editable: true },
            [400, 0],
          ),
          node(
            "approved",
            AGENT_OUTPUT_BLOCK_ID,
            { name: "approved" },
            [800, 0],
          ),
          node(
            "rejected",
            AGENT_OUTPUT_BLOCK_ID,
            { name: "rejected" },
            [800, 300],
          ),
        ],
        links: [
          link("note", "result", "review", "data"),
          link("review", "approved_data", "approved", "value"),
          link("review", "rejected_data", "rejected", "value"),
        ],
      },
    },
  });
  expect(response.ok(), `creating the agent: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  const libraryAgent = await readLibraryAgent(page, graph.id);
  expect(
    libraryAgent.has_human_in_the_loop,
    "the Library should know the agent has a review step",
  ).toBe(true);
  return {
    graphID: graph.id,
    graphVersion: graph.version,
    libraryAgentID: libraryAgent.id,
  };
}

function node(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  [x, y]: [number, number],
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x, y } },
    input_links: [],
    output_links: [],
  };
}

function link(source: string, output: string, sink: string, input: string) {
  return {
    source_id: source,
    source_name: output,
    sink_id: sink,
    sink_name: input,
  };
}

async function readLibraryAgent(
  page: Page,
  graphID: string,
): Promise<LibraryAgent> {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(response.ok(), `reading the Library agent: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as LibraryAgent;
}

// A new agent's page settles on its empty state once its run list has loaded.
// The placeholder shown until then has a Workflow Settings button of its own,
// whose dialog would close when the empty state replaces it.
async function openNewAgent(page: Page, agent: Agent) {
  await page.goto(`/library/agents/${agent.libraryAgentID}`);
  await waitForAgentPageLoad(page);
  await expect(
    page.getByRole("button", { name: "Setup your task" }),
  ).toBeVisible();
}

// Starts the run the way a person does, from the agent's empty Library page,
// and leaves the page on the run view it opens.
async function startRunInLibrary(page: Page, agent: Agent, note: string) {
  await openNewAgent(page, agent);
  await page.getByRole("button", { name: "Setup your task" }).click();
  const runDialog = page.getByRole("dialog", { name: "Run Workflow" });
  await runDialog.getByTestId("agent-input-note").fill(note);
  const started = page.waitForResponse(isRunStart(agent.graphID));
  await runDialog
    .getByRole("button", { name: "Start Task", exact: true })
    .click();
  // An agent with a review step warns before its first run, which starts
  // only once the warning is read.
  const safetyChecks = page.getByRole("dialog", { name: "Safety Checks" });
  await expect(
    safetyChecks,
    "the first run of an agent with a review step should warn first",
  ).toBeVisible();
  await safetyChecks.getByRole("button", { name: "Got it" }).click();
  const response = await started;
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  const sent = response.request().postDataJSON() as {
    inputs: Record<string, unknown>;
    dry_run?: boolean;
  };
  expect(sent.inputs, "the run should start with the note").toMatchObject({
    note,
  });
  expect(sent.dry_run ?? false, "the run should be real, not simulated").toBe(
    false,
  );
  const { id } = (await response.json()) as { id: string };
  await expect
    .poll(() => getActiveItemId(page), {
      message: "the run view should open on the new run",
    })
    .toBe(id);
  return id;
}

async function startRun(page: Page, agent: Agent, note: string) {
  const response = await page.request.post(
    `/api/proxy/api/graphs/${agent.graphID}/execute/${agent.graphVersion}`,
    { data: { inputs: { note }, credentials_inputs: {} } },
  );
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  return ((await response.json()) as { id: string }).id;
}

// Runs start INCOMPLETE, then QUEUED, so wait for a status the run rests in.
async function waitForRun(
  page: Page,
  agent: Agent,
  runID: string,
  until: RegExp,
  message: string,
): Promise<Run> {
  await expect
    .poll(async () => (await readRun(page, agent, runID)).status, {
      message,
      timeout: 30_000,
    })
    .toMatch(until);
  return readRun(page, agent, runID);
}

async function readRun(page: Page, agent: Agent, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${agent.graphID}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as Run;
}

async function waitForReview(
  page: Page,
  agent: Agent,
  runID: string,
  note: string,
) {
  const run = await waitForRun(
    page,
    agent,
    runID,
    RESTING,
    "the run should pause for its review",
  );
  expect(
    run.status,
    "the run should wait for the review, not run past it",
  ).toBe("REVIEW");
  const response = await page.request.get(
    `/api/proxy/api/review/execution/${runID}`,
  );
  expect(response.ok(), `listing the reviews: ${response.status()}`).toBe(true);
  const reviews = (await response.json()) as PendingReview[];
  expect(
    reviews,
    "the run should wait on one editable review of the note",
  ).toMatchObject([
    {
      graph_exec_id: runID,
      payload: note,
      instructions: REVIEW_NAME,
      editable: true,
      status: "WAITING",
    },
  ]);
  return reviews[0];
}

// The run view's label for one output, the one assertRunOutputValue reads.
function outputLabel(page: Page, name: string) {
  return page
    .locator("p.capitalize:visible")
    .filter({ hasText: new RegExp(`^${name}$`, "i") });
}

// A text payload is reviewed in a textarea named "Value".
function reviewDraft(page: Page) {
  return page.getByRole("textbox", { name: "Value", exact: true });
}

async function decide(page: Page, action: "Approve" | "Reject") {
  const answered = page.waitForResponse(isReviewAnswer);
  await page.getByRole("button", { name: action, exact: true }).click();
  const response = await answered;
  expect(response.ok(), `answering the review: ${response.status()}`).toBe(
    true,
  );
  return {
    sent: response.request().postDataJSON() as unknown,
    answer: (await response.json()) as unknown,
  };
}

// The switch has no name of its own: it is the one in its label's row.
function safeModeSwitch(settings: Locator) {
  return settings
    .getByText("Human-in-the-loop approval", { exact: true })
    .locator("xpath=ancestor::div[.//*[@role='switch']][1]")
    .getByRole("switch");
}

function isRunStart(graphID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${graphID}/execute/`,
    );
}

function isReviewAnswer(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/api/review/action")
  );
}

function isSettingsSave(graphID: string) {
  return (response: Response) =>
    response.request().method() === "PATCH" &&
    new URL(response.url()).pathname.endsWith(
      `/api/graphs/${graphID}/settings`,
    );
}
