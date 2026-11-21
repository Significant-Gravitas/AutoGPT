import type { APIResponse, Page, Response, TestInfo } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import {
  assertRunOutputValue,
  getActiveItemId,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { expect, test } from "./fixtures";
import { authHeaders, type JourneyUser } from "./journey-user";

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const FILL_TEXT_TEMPLATE_BLOCK_ID = "db7d8f02-2f44-4c55-ab7a-eae0941f0c30";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const OUTPUT_NAME = "reply";
const TEMPLATE_DESCRIPTION = "Made by the templates journey";
// Runs start INCOMPLETE, then QUEUED, so wait for a status a run ends in.
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

// Each version of the journey's workflow words its reply its own way, so a
// run's output says which version ran it.
const ORIGINAL_REPLY = "Reply to {{ note }}";
const EDITED_REPLY = "Reply to {{ note }}, from the edited workflow";

interface Agent {
  name: string;
  graphID: string;
  graphVersion: number;
  libraryAgentID: string;
}

interface Run {
  status: string;
  graph_version: number;
  preset_id: string | null;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown[]>;
}

interface Template {
  id: string;
  name: string;
  description: string;
  graph_id: string;
  graph_version: number;
  inputs: Record<string, unknown>;
}

test.describe("templates journey", () => {
  test("a run saved as a template opens with its input, keeps an edit through a reload, and starts a task that runs with the edit", async ({
    page,
    journeyUser: _user,
  }) => {
    const stamp = Date.now();
    const firstNote = `First note ${stamp}`;
    const editedNote = `Edited note ${stamp}`;
    const templateName = `Journey template ${stamp}`;
    const agent = await createReplyAgent(page);

    const firstRunID = await startRunInLibrary(page, agent, firstNote);
    const firstRun = await waitForRunToEnd(page, agent, firstRunID);
    expect(firstRun.status).toBe("COMPLETED");
    expect(firstRun.outputs).toEqual({
      [OUTPUT_NAME]: [reply(ORIGINAL_REPLY, firstNote)],
    });

    const template = await saveRunAsTemplate(page, firstRunID, templateName);
    expect(
      template,
      "the template should keep the run's input and workflow version",
    ).toMatchObject({
      name: templateName,
      description: TEMPLATE_DESCRIPTION,
      graph_id: agent.graphID,
      graph_version: agent.graphVersion,
      inputs: { note: firstNote },
    });

    await openTemplateFromTab(page, template);
    await expectTemplateShows(page, {
      name: templateName,
      description: TEMPLATE_DESCRIPTION,
      note: firstNote,
    });

    await templateField(page, "note").fill(editedNote);
    const { sent, saved } = await saveTemplateChanges(page, template.id);
    expect(sent, "saving should send the edited input").toMatchObject({
      inputs: { note: editedNote },
    });
    expect(saved.inputs, "the saved template should have the edit").toEqual({
      note: editedNote,
    });

    await page.reload();
    await waitForAgentPageLoad(page);
    await expect(
      templateField(page, "note"),
      "the edit should survive a reload",
    ).toHaveValue(editedNote);
    expect(
      (await readTemplate(page, template.id)).inputs,
      "the backend should keep the edit",
    ).toEqual({ note: editedNote });

    const runID = await startTaskFromTemplate(page, template.id);
    const run = await waitForRunToEnd(page, agent, runID);
    expect(run.status).toBe("COMPLETED");
    expect(
      run,
      "the task should run the edited template, not the run it came from",
    ).toMatchObject({
      preset_id: template.id,
      graph_version: template.graph_version,
      inputs: { note: editedNote },
      outputs: { [OUTPUT_NAME]: [reply(ORIGINAL_REPLY, editedNote)] },
    });
    await assertRunOutputValue(
      page,
      OUTPUT_NAME,
      reply(ORIGINAL_REPLY, editedNote),
    );

    // Newest first, in the API as in the Library.
    const newestFirst = [runID, firstRunID];
    expect(await listRunIDs(page, agent)).toEqual(newestFirst);
    await expect
      .poll(() => taskListIDs(page), {
        message: "the Tasks list should show the template's task first",
      })
      .toEqual(newestFirst);
  });

  test("after its workflow is edited, a template starts a task on the version it stores, with that version's output", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const stamp = Date.now();
    const note = `Versioned note ${stamp}`;
    const agent = await createReplyAgent(page);
    const sourceRunID = await startRun(page, agent, note);
    expect((await waitForRunToEnd(page, agent, sourceRunID)).status).toBe(
      "COMPLETED",
    );
    const template = await createTemplate(
      page,
      sourceRunID,
      `Versioned template ${stamp}`,
    );
    expect(template.graph_version).toBe(agent.graphVersion);

    // Edited after the template was saved: the new version words the reply
    // differently, and becomes the version the Library runs.
    const editedVersion = await saveEditedVersion(page, agent);
    expect(editedVersion).toBe(agent.graphVersion + 1);
    const replyOf: Record<number, string> = {
      [agent.graphVersion]: reply(ORIGINAL_REPLY, note),
      [editedVersion]: reply(EDITED_REPLY, note),
    };
    const libraryVersion = (await readLibraryAgent(page, agent.graphID))
      .graph_version;
    const stored = await readTemplate(page, template.id);
    observe(
      testInfo,
      `after the edit to v${editedVersion}, the template stores ` +
        `v${stored.graph_version} and the Library agent is on v${libraryVersion}`,
    );

    // The invariant, whichever version a template should follow (#17201):
    // its task runs the version it stores, with that version's output.
    await openTemplate(page, agent, template.id);
    await expect(templateField(page, "note")).toHaveValue(note);
    const runID = await startTaskFromTemplate(page, template.id);
    const run = await waitForRunToEnd(page, agent, runID);
    observe(
      testInfo,
      `Start task from template ran v${run.graph_version} ` +
        `(${run.status}) and output ${JSON.stringify(run.outputs)}`,
    );
    expect(run.status, "the template's task should complete").toBe("COMPLETED");
    expect(
      run.graph_version,
      "the task should run the version the template stores",
    ).toBe(stored.graph_version);
    expect(
      Object.keys(replyOf),
      `the task ran v${run.graph_version}, which this journey didn't make`,
    ).toContain(String(run.graph_version));
    expect(
      run.outputs,
      "the task's output should be the one of the version it ran",
    ).toEqual({ [OUTPUT_NAME]: [replyOf[run.graph_version]] });
    await assertRunOutputValue(page, OUTPUT_NAME, replyOf[run.graph_version]);

    // The page's other way to start from the template: New workflow task
    // opens the run dialog with the template's input filled in.
    await openTemplate(page, agent, template.id);
    await expect(templateField(page, "note")).toHaveValue(note);
    const dialogRunID = await startTaskFromRunDialog(page, agent, note);
    const dialogRun = await waitForRunToEnd(page, agent, dialogRunID);
    observe(
      testInfo,
      `New workflow task, filled in from the template, ran ` +
        `v${dialogRun.graph_version} (${dialogRun.status}) and output ` +
        JSON.stringify(dialogRun.outputs),
    );
    expect(dialogRun.status).toBe("COMPLETED");
    expect(
      dialogRun.outputs,
      "the task's output should be the one of the version it ran",
    ).toEqual({ [OUTPUT_NAME]: [replyOf[dialogRun.graph_version]] });
  });

  test("a template's new name, cleared description and edited input survive a reload and signing out and back in", async ({
    page,
    journeyUser,
  }) => {
    const stamp = Date.now();
    const note = `Saved note ${stamp}`;
    const editedNote = `Kept note ${stamp}`;
    const renamed = `Renamed template ${stamp}`;
    const agent = await createReplyAgent(page);
    const runID = await startRun(page, agent, note);
    expect((await waitForRunToEnd(page, agent, runID)).status).toBe(
      "COMPLETED",
    );
    const template = await createTemplate(
      page,
      runID,
      `Journey template ${stamp}`,
    );

    await openTemplate(page, agent, template.id);
    await expectTemplateShows(page, {
      name: template.name,
      description: TEMPLATE_DESCRIPTION,
      note,
    });
    await templateField(page, "Name").fill(renamed);
    await templateField(page, "Description").clear();
    await templateField(page, "note").fill(editedNote);
    const { sent, saved } = await saveTemplateChanges(page, template.id);
    const edited = {
      name: renamed,
      description: "",
      inputs: { note: editedNote },
    };
    expect(sent, "saving should send every edit").toMatchObject(edited);
    expect(saved, "the saved template should have every edit").toMatchObject(
      edited,
    );

    const shown = { name: renamed, description: "", note: editedNote };
    await page.reload();
    await waitForAgentPageLoad(page);
    await expectTemplateShows(page, shown);

    await signOutAndBackIn(page, journeyUser, template.id);
    await page.goto(`/library/agents/${agent.libraryAgentID}`);
    await waitForAgentPageLoad(page);
    await openTemplateFromTab(page, { id: template.id, name: renamed });
    await expectTemplateShows(page, shown);
    expect(
      await readTemplate(page, template.id),
      "the backend should keep every edit",
    ).toMatchObject(edited);
  });
});

// AgentInput "note" -> Fill Text Template, worded by `format` -> AgentOutput.
function replyGraph(name: string, format: string) {
  return {
    name,
    description: "Replies to a note, for the templates journey",
    is_active: true,
    nodes: [
      node("note", AGENT_INPUT_BLOCK_ID, { name: "note" }, [0, 0]),
      node(
        "compose",
        FILL_TEXT_TEMPLATE_BLOCK_ID,
        { format, values: {} },
        [400, 0],
      ),
      node("reply", AGENT_OUTPUT_BLOCK_ID, { name: OUTPUT_NAME }, [800, 0]),
    ],
    links: [
      link("note", "result", "compose", "values_#_note"),
      link("compose", "output", "reply", "value"),
    ],
  };
}

function reply(format: string, note: string) {
  return format.replace("{{ note }}", note);
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

async function createReplyAgent(page: Page): Promise<Agent> {
  const name = createUniqueAgentName("Journey template");
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph: replyGraph(name, ORIGINAL_REPLY) },
  });
  expect(response.ok(), `creating the agent: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  return {
    name,
    graphID: graph.id,
    graphVersion: graph.version,
    libraryAgentID: (await readLibraryAgent(page, graph.id)).id,
  };
}

// Saves the workflow again, as the builder's Save does: a new version, which
// the Library agent moves to.
async function saveEditedVersion(page: Page, agent: Agent) {
  const response = await page.request.put(
    `/api/proxy/api/graphs/${agent.graphID}`,
    { data: { id: agent.graphID, ...replyGraph(agent.name, EDITED_REPLY) } },
  );
  expect(
    response.ok(),
    `saving the edited workflow: ${response.status()}`,
  ).toBe(true);
  return ((await response.json()) as { graph: { version: number } }).graph
    .version;
}

async function readLibraryAgent(page: Page, graphID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(response.ok(), `reading the Library agent: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as { id: string; graph_version: number };
}

// Starts a run the way a person does, from the new agent's empty Library
// page, and leaves the page on the run view it opens.
async function startRunInLibrary(page: Page, agent: Agent, note: string) {
  await page.goto(`/library/agents/${agent.libraryAgentID}`);
  await waitForAgentPageLoad(page);
  // The page settles on its empty state once its run list has loaded.
  const setUp = page.getByRole("button", { name: "Setup your task" });
  await expect(setUp).toBeVisible();
  await setUp.click();
  const runDialog = page.getByRole("dialog", { name: "Run Workflow" });
  await runDialog.getByTestId("agent-input-note").fill(note);
  const started = page.waitForResponse(isRunStart(agent.graphID));
  await runDialog
    .getByRole("button", { name: "Start Task", exact: true })
    .click();
  const id = await startedRunID(await started);
  await expect
    .poll(() => getActiveItemId(page), {
      message: "the run view should open on the new run",
    })
    .toBe(id);
  return id;
}

// With a template open, New workflow task opens the run dialog with the
// template's inputs filled in.
async function startTaskFromRunDialog(page: Page, agent: Agent, note: string) {
  await page.getByRole("button", { name: "New workflow task" }).click();
  const runDialog = page.getByRole("dialog", { name: "Run Workflow" });
  await expect(
    runDialog.getByTestId("agent-input-note"),
    "the run dialog should start with the template's input",
  ).toHaveValue(note);
  const started = page.waitForResponse(isRunStart(agent.graphID));
  await runDialog
    .getByRole("button", { name: "Start Task", exact: true })
    .click();
  const id = await startedRunID(await started);
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
  return startedRunID(response);
}

async function startedRunID(response: APIResponse | Response) {
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  return ((await response.json()) as { id: string }).id;
}

async function waitForRunToEnd(page: Page, agent: Agent, runID: string) {
  await expect
    .poll(async () => (await readRun(page, agent, runID)).status, {
      message: "the run should end",
      timeout: 30_000,
    })
    .toMatch(RUN_ENDED);
  return readRun(page, agent, runID);
}

async function readRun(page: Page, agent: Agent, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${agent.graphID}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as Run;
}

async function listRunIDs(page: Page, agent: Agent) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${agent.graphID}/executions`,
  );
  expect(response.ok(), `listing the runs: ${response.status()}`).toBe(true);
  const { executions } = (await response.json()) as {
    executions: { id: string }[];
  };
  return executions.map((run) => run.id);
}

// The runs the Tasks tab lists, top to bottom.
function taskListIDs(page: Page) {
  return page
    .getByTestId(/^library-run-item-/)
    .evaluateAll((rows) =>
      rows.map((row) =>
        row.getAttribute("data-testid")?.replace("library-run-item-", ""),
      ),
    );
}

// Save as template, from the run's own row in the Tasks list: the run view
// has a "More actions" menu of its own, and so does every other row.
async function saveRunAsTemplate(page: Page, runID: string, name: string) {
  const row = page.getByTestId(`library-run-item-${runID}`);
  // The row re-renders as its run finishes, so open the menu after that.
  await expect(row.getByRole("img", { name: "Task completed" })).toBeVisible();
  await row.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: "Save as template" }).click();
  const dialog = page.getByRole("dialog", { name: "Create Template" });
  await dialog.getByRole("textbox", { name: "Name", exact: true }).fill(name);
  await dialog
    .getByRole("textbox", { name: "Description", exact: true })
    .fill(TEMPLATE_DESCRIPTION);
  const created = page.waitForResponse(
    isPresetCall("POST", "/api/library/presets"),
  );
  await dialog.getByRole("button", { name: "Create Template" }).click();
  const response = await created;
  expect(response.ok(), `saving the template: ${response.status()}`).toBe(true);
  expect(
    response.request().postDataJSON(),
    "the template should be made from the run",
  ).toMatchObject({ name, graph_execution_id: runID });
  await expect(
    dialog,
    "the dialog should close once the template is saved",
  ).toBeHidden();
  return (await response.json()) as Template;
}

async function createTemplate(page: Page, runID: string, name: string) {
  const response = await page.request.post("/api/proxy/api/library/presets", {
    data: {
      name,
      description: TEMPLATE_DESCRIPTION,
      graph_execution_id: runID,
    },
  });
  expect(response.ok(), `saving the template: ${response.status()}`).toBe(true);
  return (await response.json()) as Template;
}

async function readTemplate(page: Page, templateID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/presets/${templateID}`,
  );
  expect(response.ok(), `reading the template: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as Template;
}

// Opens a template from the agent page's Templates tab, by its name there.
async function openTemplateFromTab(
  page: Page,
  template: Pick<Template, "id" | "name">,
) {
  const tab = page.getByRole("tab", { name: /^Templates/ });
  await expect(tab, "the Templates tab should count the template").toHaveText(
    /^Templates\s*1$/,
  );
  await tab.click();
  await page
    .getByRole("tabpanel", { name: /^Templates/ })
    .getByText(template.name, { exact: true })
    .click();
  await expect
    .poll(() => activeView(page), {
      message: "the template should open in the Templates tab",
    })
    .toEqual({ tab: "templates", item: template.id });
}

async function openTemplate(page: Page, agent: Agent, templateID: string) {
  await page.goto(
    `/library/agents/${agent.libraryAgentID}?activeTab=templates&activeItem=${templateID}`,
  );
  await waitForAgentPageLoad(page);
}

// The open template's fields: its name and description, then the agent's
// inputs, each named after its input.
function templateField(page: Page, name: "Name" | "Description" | "note") {
  return page.getByRole("textbox", { name, exact: true });
}

async function expectTemplateShows(
  page: Page,
  shown: { name: string; description: string; note: string },
) {
  await expect(templateField(page, "Name")).toHaveValue(shown.name);
  await expect(templateField(page, "Description")).toHaveValue(
    shown.description,
  );
  await expect(templateField(page, "note")).toHaveValue(shown.note);
}

async function saveTemplateChanges(page: Page, templateID: string) {
  const saved = page.waitForResponse(
    isPresetCall("PATCH", `/api/library/presets/${templateID}`),
  );
  await page.getByRole("button", { name: "Save changes" }).click();
  const response = await saved;
  expect(
    response.ok(),
    `saving the template's changes: ${response.status()}`,
  ).toBe(true);
  return {
    sent: response.request().postDataJSON() as Record<string, unknown>,
    saved: (await response.json()) as Template,
  };
}

// Starts a task from the open template, which moves the page to that task.
async function startTaskFromTemplate(page: Page, templateID: string) {
  const started = page.waitForResponse(
    isPresetCall("POST", `/api/library/presets/${templateID}/execute`),
  );
  await page.getByRole("button", { name: "Start task from template" }).click();
  const id = await startedRunID(await started);
  await expect
    .poll(() => activeView(page), {
      message: "starting a task should open it on the Tasks tab",
    })
    .toEqual({ tab: "runs", item: id });
  return id;
}

// Ends the session and starts a new one, with no page of the app open
// meanwhile: the /logout page's own requests race its sign-out, and one that
// loses answers 401 (session-resilience-journey covers that page).
async function signOutAndBackIn(
  page: Page,
  user: JourneyUser,
  templateID: string,
) {
  await page.goto("about:blank");
  const signedOut = await page.request.post("/api/auth/sign-out", {
    headers: authHeaders(),
    data: {},
  });
  expect(signedOut.ok(), `signing out: ${signedOut.status()}`).toBe(true);
  const refused = await page.request.get(
    `/api/proxy/api/library/presets/${templateID}`,
  );
  expect(
    refused.status(),
    "signed out, the template should be out of reach",
  ).toBe(401);
  const signedIn = await page.request.post("/api/auth/sign-in/email", {
    headers: authHeaders(),
    data: { email: user.email, password: user.password },
  });
  expect(signedIn.ok(), `signing back in: ${signedIn.status()}`).toBe(true);
}

function activeView(page: Page) {
  const params = new URL(page.url()).searchParams;
  return { tab: params.get("activeTab"), item: params.get("activeItem") };
}

function isPresetCall(method: string, path: string) {
  return (response: Response) =>
    response.request().method() === method &&
    new URL(response.url()).pathname.endsWith(path);
}

function isRunStart(graphID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${graphID}/execute/`,
    );
}

// What the journey saw, in the report: details worth reading next to the
// assertions, such as which version each way of starting a template ran.
function observe(testInfo: TestInfo, description: string) {
  testInfo.annotations.push({ type: "observed", description });
}
