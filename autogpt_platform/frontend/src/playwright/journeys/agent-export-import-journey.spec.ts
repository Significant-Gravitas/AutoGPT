import { randomUUID } from "crypto";
import { readFile } from "fs/promises";
import type { Download, Page, Response, TestInfo } from "@playwright/test";
import { BuildPage, createUniqueAgentName } from "../pages/build.page";
import {
  assertRunOutputValue,
  getActiveItemId,
  LibraryPage,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { expect, test } from "./fixtures";

const BLOCK = {
  input: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  output: "363ae599-353e-4804-937e-b2ee3cef3da4",
  fillTextTemplate: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
  createList: "a912d5c7-6e00-4542-b2a9-8034136930e4",
  subWorkflow: "e189baac-8c20-45a1-94a7-55177ea42565",
  textToSpeech: "4ff1ff6d-cc40-4caa-ae69-011daa20c378",
} as const;

// How the comparison names each kind of step, in place of its block ID.
const STEP_KINDS: Record<string, string> = {
  [BLOCK.input]: "Agent Input",
  [BLOCK.output]: "Agent Output",
  [BLOCK.fillTextTemplate]: "Fill Text Template",
  [BLOCK.createList]: "Create List",
  [BLOCK.subWorkflow]: "Sub-workflow",
  [BLOCK.textToSpeech]: "Text to Speech",
};

// People word their workflows in their own language, so this one carries
// accents, typographic punctuation and symbols in its name, an input's
// default, its templates and a step's name.
const DESCRIPTION = "Écrit une note, pour le parcours d’export";
const SIGN_OFF = "Bien à vous ✓";
const GREETING = "Grüße";
const COMPOSE_FORMAT =
  "{{ greeting }} — {{ topic }}: {{ lines | join(' · ') }}";
const COMPOSE_STEP_NAME = "Compose the note ✎";
const FRAME_FORMAT = "« {{ line }} »";
const OUTPUT_NAME = "note";
const VOICE = "Scarlett";
// Runs start INCOMPLETE, then QUEUED, so wait for a status a run ends in.
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

interface StepJSON {
  id: string;
  block_id: string;
  input_default: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

interface LinkJSON {
  source_id: string;
  source_name: string;
  sink_id: string;
  sink_name: string;
  is_static?: boolean;
}

interface IOSchema {
  properties: Record<string, unknown>;
  required?: string[];
}

// A workflow as the API returns it and as an export file holds it.
interface GraphJSON {
  id: string;
  version: number;
  user_id?: string;
  name: string;
  description: string;
  nodes: StepJSON[];
  links: LinkJSON[];
  input_schema: IOSchema;
  output_schema: IOSchema;
  sub_graphs?: GraphJSON[];
}

interface Workflow {
  graph: GraphJSON;
  libraryAgentID: string;
}

interface Credential {
  id: string;
  provider: string;
  type: string;
  title: string;
}

test.describe("workflow export and import journey", () => {
  test("a workflow exported from the Library and imported back runs to the same output, and keeps its inputs, links, step names and sub-workflow", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    // Two Library runs, each of which starts a sub-workflow run, and an
    // import that opens the builder, which can take over 10 s to draw
    // (#16927): more than the default 90 s allows on a slow runner.
    test.setTimeout(120_000);
    const topic = `Topic ${Date.now()}`;
    const original = await createNoteWorkflow(page);

    const originalRun = await runInLibrary(page, original, topic);
    expect(
      originalRun.offeredSignOff,
      "the run dialog should offer the sign-off's default",
    ).toBe(SIGN_OFF);
    expect(originalRun.result, "the workflow should run as built").toEqual({
      status: "COMPLETED",
      outputs: { [OUTPUT_NAME]: [note(topic)] },
    });
    await assertRunOutputValue(page, OUTPUT_NAME, note(topic));

    const saved = await readGraph(page, original.graph.id);
    const exported = await exportFromRunView(page, testInfo);
    expect
      .soft(
        differences(shapeOf(saved), shapeOf(exported.graph)),
        "the exported file should hold the workflow as saved",
      )
      .toEqual([]);

    const imported = await importInLibrary(page, exported.path);
    expect
      .soft(
        imported.filledIn,
        "the import dialog should fill in the file's name and description",
      )
      .toEqual({ name: saved.name, description: saved.description });
    const importedGraph = await readGraph(page, imported.graphID);
    await new BuildPage(page).waitForNodeOnCanvas(importedGraph.nodes.length);
    observe(
      testInfo,
      `saved v${saved.version}, exported as "${exported.suggestedName}" ` +
        `holding v${exported.graph.version}, imported as v${importedGraph.version}`,
    );

    const importedRun = await runInLibrary(
      page,
      {
        graph: importedGraph,
        libraryAgentID: await libraryAgentID(page, importedGraph.id),
      },
      topic,
    );
    expect
      .soft(
        importedRun.offeredSignOff,
        "the imported workflow's run dialog should offer the same default",
      )
      .toBe(SIGN_OFF);
    // The run is the check that matters; the comparison says what broke it.
    const changes = differences(shapeOf(saved), shapeOf(importedGraph));
    expect(
      importedRun.result,
      "the imported workflow should run as the original did. The import " +
        `changed:\n${listed(changes)}`,
    ).toEqual(originalRun.result);
    await assertRunOutputValue(page, OUTPUT_NAME, note(topic));
    expect(
      changes,
      "the imported workflow should be the one exported, ids and versions aside",
    ).toEqual([]);
  });

  test("an exported workflow carries none of its owner's credentials, in its own steps or its sub-workflow's", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const apiKey = `journey-key-${randomUUID()}`;
    const credential = await createAPIKey(page, apiKey);
    const workflow = await createSpeakingWorkflow(page, credential);
    const saved = await readGraph(page, workflow.graph.id);
    expect(
      credentialIDs(saved),
      "the saved workflow should use the key in its own step and its sub-workflow's",
    ).toEqual([credential.id, credential.id]);

    await page.goto(`/library/agents/${workflow.libraryAgentID}`);
    await waitForAgentPageLoad(page);
    // A workflow with no tasks yet offers its export beside its details.
    await expect(
      page.getByRole("button", { name: "Setup your task" }),
    ).toBeVisible();
    const exported = await exportFromDetails(page, testInfo);

    expect(
      exported.text,
      "the file should not name the owner's key",
    ).not.toContain(credential.id);
    expect(exported.text, "the file should not hold the key").not.toContain(
      apiKey,
    );
    expect(
      saved.user_id,
      "the saved workflow should name its owner",
    ).toBeTruthy();
    expect(exported.text, "the file should not name its owner").not.toContain(
      String(saved.user_id),
    );
    expect(
      speakingSteps(exported.graph).map((step) => step.input_default),
      "the file should keep both speaking steps, without their key",
    ).toEqual([{ voice_id: VOICE }, { voice_id: VOICE }]);
  });
});

// "topic" and "sign_off" (which has a default) -> a list of both, through
// list pins -> a template that takes the topic and that list through dict
// pins, beside a greeting set on the step itself -> a sub-workflow that
// frames the line -> the "note" output.
async function createNoteWorkflow(page: Page): Promise<Workflow> {
  const frame = await saveWorkflow(page, {
    name: createUniqueAgentName("Journey frame"),
    description: "Frames a line, for the export journey",
    nodes: [
      step("line", BLOCK.input, { name: "line", title: "Line" }, [0, 0]),
      step(
        "frame",
        BLOCK.fillTextTemplate,
        { format: FRAME_FORMAT, values: {} },
        [400, 0],
      ),
      step(
        "framed",
        BLOCK.output,
        { name: "framed", title: "Framed" },
        [800, 0],
      ),
    ],
    links: [
      link("line", "result", "frame", "values_#_line"),
      link("frame", "output", "framed", "value"),
    ],
  });
  const graph = await saveWorkflow(page, {
    name: createUniqueAgentName("Résumé du parcours"),
    description: DESCRIPTION,
    nodes: [
      step(
        "topic",
        BLOCK.input,
        {
          name: "topic",
          title: "Topic",
          description: "What the note is about",
        },
        [0, 0],
      ),
      step(
        "sign_off",
        BLOCK.input,
        {
          name: "sign_off",
          title: "Sign-off",
          description: "How the note ends — leave it as it is",
          value: SIGN_OFF,
        },
        [0, 320],
      ),
      step("lines", BLOCK.createList, {}, [400, 320]),
      step(
        "compose",
        BLOCK.fillTextTemplate,
        { format: COMPOSE_FORMAT, values: { greeting: GREETING } },
        [800, 0],
        { customized_name: COMPOSE_STEP_NAME },
      ),
      step("frame", BLOCK.subWorkflow, runs(frame), [1200, 0]),
      step(
        "note",
        BLOCK.output,
        { name: OUTPUT_NAME, title: "Note", description: "The framed note" },
        [1600, 0],
      ),
    ],
    links: [
      link("topic", "result", "lines", "values_$_0"),
      link("sign_off", "result", "lines", "values_$_1"),
      link("topic", "result", "compose", "values_#_topic"),
      link("lines", "list", "compose", "values_#_lines"),
      link("compose", "output", "frame", "line"),
      link("frame", "framed", "note", "value"),
    ],
  });
  return { graph, libraryAgentID: await libraryAgentID(page, graph.id) };
}

function note(topic: string) {
  return `« ${GREETING} — ${topic}: ${topic} · ${SIGN_OFF} »`;
}

// Speaks its text with the user's key, in a step of its own and in a step
// of its sub-workflow. Nothing runs it: the test only exports it.
async function createSpeakingWorkflow(
  page: Page,
  credential: Credential,
): Promise<Workflow> {
  const key = {
    id: credential.id,
    provider: credential.provider,
    type: credential.type,
    title: credential.title,
  };
  const speak = await saveWorkflow(page, {
    name: createUniqueAgentName("Journey speak"),
    description: "Speaks its text, for the export journey",
    nodes: [
      step("text", BLOCK.input, { name: "text", title: "Text" }, [0, 0]),
      step(
        "speak",
        BLOCK.textToSpeech,
        { voice_id: VOICE, credentials: key },
        [400, 0],
      ),
      step("speech", BLOCK.output, { name: "speech" }, [800, 0]),
    ],
    links: [
      link("text", "result", "speak", "text"),
      link("speak", "mp3_url", "speech", "value"),
    ],
  });
  const graph = await saveWorkflow(page, {
    name: createUniqueAgentName("Journey keys"),
    description: "Speaks its text twice, for the export journey",
    nodes: [
      step("text", BLOCK.input, { name: "text", title: "Text" }, [0, 0]),
      step(
        "speak",
        BLOCK.textToSpeech,
        { voice_id: VOICE, credentials: key },
        [400, 0],
      ),
      step("speech", BLOCK.output, { name: "speech" }, [800, 0]),
      step("again", BLOCK.subWorkflow, runs(speak), [400, 320]),
      step("speech_again", BLOCK.output, { name: "speech_again" }, [800, 320]),
    ],
    links: [
      link("text", "result", "speak", "text"),
      link("speak", "mp3_url", "speech", "value"),
      link("text", "result", "again", "text"),
      link("again", "speech", "speech_again", "value"),
    ],
  });
  return { graph, libraryAgentID: await libraryAgentID(page, graph.id) };
}

// A sub-workflow step's input, as the builder writes it.
function runs(graph: GraphJSON) {
  return {
    user_id: "",
    graph_id: graph.id,
    graph_version: graph.version,
    agent_name: graph.name,
    inputs: {},
    input_schema: graph.input_schema,
    output_schema: graph.output_schema,
  };
}

function step(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  [x, y]: [number, number],
  metadata: Record<string, unknown> = {},
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x, y }, ...metadata },
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

// Saved as the builder saves: a workflow saved as an upload loses its steps
// that run one of the user's own workflows (clear_sub_graph_refs_owned_by).
async function saveWorkflow(
  page: Page,
  graph: Pick<GraphJSON, "name" | "description"> & {
    nodes: ReturnType<typeof step>[];
    links: ReturnType<typeof link>[];
  },
) {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "builder", graph: { ...graph, is_active: true } },
  });
  expect(
    response.ok(),
    `saving "${graph.name}": ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as GraphJSON;
}

async function readGraph(page: Page, graphID: string) {
  const response = await page.request.get(`/api/proxy/api/graphs/${graphID}`);
  expect(response.ok(), `reading the workflow: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as GraphJSON;
}

async function libraryAgentID(page: Page, graphID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(response.ok(), `reading the Library agent: ${response.status()}`).toBe(
    true,
  );
  return ((await response.json()) as { id: string }).id;
}

async function createAPIKey(page: Page, apiKey: string): Promise<Credential> {
  const response = await page.request.post(
    "/api/proxy/api/integrations/unreal_speech/credentials",
    {
      data: {
        type: "api_key",
        provider: "unreal_speech",
        title: `Journey key ${Date.now()}`,
        api_key: apiKey,
      },
    },
  );
  expect(response.ok(), `saving the API key: ${response.status()}`).toBe(true);
  return (await response.json()) as Credential;
}

// Runs the workflow the way a person does, from its new Library page, with
// `topic` typed in and the sign-off left at what the dialog offers, and
// waits for the run to end.
async function runInLibrary(page: Page, workflow: Workflow, topic: string) {
  await page.goto(`/library/agents/${workflow.libraryAgentID}`);
  await waitForAgentPageLoad(page);
  // The page settles on its empty state once its run list has loaded.
  const setUp = page.getByRole("button", { name: "Setup your task" });
  await expect(setUp).toBeVisible();
  await setUp.click();
  const runDialog = page.getByRole("dialog", { name: "Run Workflow" });
  await runDialog.getByTestId("agent-input-topic").fill(topic);
  const signOff = runDialog.getByTestId("agent-input-sign_off");
  await expect(signOff).toBeVisible();
  const offeredSignOff = await signOff.inputValue();
  const started = page.waitForResponse(isRunStart(workflow.graph.id));
  await runDialog
    .getByRole("button", { name: "Start Task", exact: true })
    .click();
  const response = await started;
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  const runID = ((await response.json()) as { id: string }).id;
  await expect
    .poll(() => getActiveItemId(page), {
      message: "the run view should open on the new run",
    })
    .toBe(runID);
  await expect
    .poll(async () => (await readRun(page, workflow, runID)).status, {
      message: "the run should end",
      timeout: 30_000,
    })
    .toMatch(RUN_ENDED);
  const { status, outputs } = await readRun(page, workflow, runID);
  return { offeredSignOff, result: { status, outputs } };
}

async function readRun(page: Page, workflow: Workflow, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graph.id}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as {
    status: string;
    outputs: Record<string, unknown[]>;
  };
}

// Each run in the sidebar has a "More actions" menu too; the run view's own
// sits beside its Share control.
async function exportFromRunView(page: Page, testInfo: TestInfo) {
  await page
    .getByRole("button", { name: "Share results" })
    .locator("xpath=ancestor::div[.//button[@aria-label='More actions']][1]")
    .getByRole("button", { name: "More actions" })
    .click();
  const download = page.waitForEvent("download");
  await page.getByRole("menuitem", { name: "Export workflow to file" }).click();
  return keepExport(await download, testInfo);
}

async function exportFromDetails(page: Page, testInfo: TestInfo) {
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export workflow to file" }).click();
  return keepExport(await download, testInfo);
}

async function keepExport(download: Download, testInfo: TestInfo) {
  const path = testInfo.outputPath(`export-${randomUUID()}.json`);
  await download.saveAs(path);
  const text = await readFile(path, "utf8");
  return {
    path,
    text,
    graph: JSON.parse(text) as GraphJSON,
    suggestedName: download.suggestedFilename(),
  };
}

// Imports the file through the Library's Import dialog, keeping the name and
// description it fills in from the file. The import opens the new workflow
// in the builder.
async function importInLibrary(page: Page, filePath: string) {
  await page.goto("/library");
  await new LibraryPage(page).openUploadDialog();
  const dialog = page.getByRole("dialog", { name: "Import" });
  await dialog.locator('input[type="file"]').setInputFiles(filePath);
  const name = dialog.getByRole("textbox", { name: "Workflow name" });
  await expect(name, "the dialog should read the file").not.toHaveValue("");
  const filledIn = {
    name: await name.inputValue(),
    description: await dialog
      .getByRole("textbox", { name: "Workflow description" })
      .inputValue(),
  };
  const created = page.waitForResponse(isGraphCreation);
  await dialog.getByRole("button", { name: "Upload", exact: true }).click();
  const response = await created;
  expect(response.ok(), `importing the file: ${response.status()}`).toBe(true);
  // The page leaves for the builder as soon as the import is saved, which
  // drops the response's body, so the new workflow's id comes from the URL.
  await expect(page).toHaveURL(/\/build\?.*flowID=/);
  const graphID = new URL(page.url()).searchParams.get("flowID");
  expect(graphID, "the builder should open the imported workflow").toBeTruthy();
  return { graphID: String(graphID), filledIn };
}

function isGraphCreation(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/api/graphs")
  );
}

function isRunStart(graphID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${graphID}/execute/`,
    );
}

// What a round trip must keep of a workflow: its steps, keyed by kind and
// input or output name, with their inputs and metadata (a step's custom name
// and position); its links between them; its inputs' and outputs' schemas;
// and the same for each sub-workflow it carries. IDs and versions differ by
// design, so a sub-workflow step names the workflow it runs instead.
function shapeOf(graph: GraphJSON) {
  const carried = graph.sub_graphs ?? [];
  return {
    ...workflowShape(graph, carried),
    sub_workflows: Object.fromEntries(
      carried.map((sub) => [sub.name, workflowShape(sub, carried)]),
    ),
  };
}

function workflowShape(graph: GraphJSON, carried: GraphJSON[]) {
  const keys = stepKeys(graph.nodes);
  return {
    name: graph.name,
    description: graph.description,
    steps: Object.fromEntries(
      graph.nodes.map((node) => [
        keys.get(node.id),
        {
          block_id: node.block_id,
          input_default: comparableInput(node, carried),
          metadata: node.metadata,
        },
      ]),
    ),
    links: Object.fromEntries(
      graph.links.map((l) => [
        `${keys.get(l.source_id)}.${l.source_name} → ${keys.get(l.sink_id)}.${l.sink_name}`,
        { is_static: l.is_static ?? false },
      ]),
    ),
    inputs: graph.input_schema.properties,
    required_inputs: [...(graph.input_schema.required ?? [])].sort(),
    outputs: graph.output_schema.properties,
  };
}

// Each step's key, by node ID. A second step with the same key gets "#2", so
// a step the round trip added shows up as one the original doesn't have.
function stepKeys(nodes: StepJSON[]) {
  const seen = new Map<string, number>();
  return new Map(
    nodes.map((node) => {
      const key = stepKey(node);
      const count = (seen.get(key) ?? 0) + 1;
      seen.set(key, count);
      return [node.id, count === 1 ? key : `${key} #${count}`];
    }),
  );
}

function stepKey(node: StepJSON) {
  const kind = STEP_KINDS[node.block_id] ?? node.block_id;
  const name = node.input_default.name;
  return typeof name === "string" ? `${kind} "${name}"` : kind;
}

function comparableInput(node: StepJSON, carried: GraphJSON[]) {
  if (node.block_id !== BLOCK.subWorkflow) return node.input_default;
  const {
    graph_id: graphID,
    graph_version: _version,
    user_id: _owner,
    ...input
  } = node.input_default;
  const target = carried.find((sub) => sub.id === graphID);
  return {
    ...input,
    runs: target?.name ?? (graphID ? "a workflow it doesn't carry" : "nothing"),
  };
}

function credentialIDs(graph: GraphJSON) {
  return allSteps(graph)
    .map((node) => node.input_default.credentials as { id?: string } | null)
    .flatMap((credentials) => (credentials?.id ? [credentials.id] : []));
}

function speakingSteps(graph: GraphJSON) {
  return allSteps(graph).filter((node) => node.block_id === BLOCK.textToSpeech);
}

function allSteps(graph: GraphJSON) {
  return [graph, ...(graph.sub_graphs ?? [])].flatMap((g) => g.nodes);
}

// Each value that differs, as "path: before → after".
function differences(
  before: unknown,
  after: unknown,
  path: string[] = [],
): string[] {
  if (isRecord(before) && isRecord(after)) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
    return [...keys]
      .sort()
      .flatMap((key) => differences(before[key], after[key], [...path, key]));
  }
  if (JSON.stringify(before) === JSON.stringify(after)) return [];
  return [`${path.join(" › ")}: ${shown(before)} → ${shown(after)}`];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// JSON, with invisible control characters spelled out.
function shown(value: unknown) {
  if (value === undefined) return "(missing)";
  return JSON.stringify(value).replace(
    /[\u007f-\u009f]/g,
    (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

function listed(changes: string[]) {
  if (changes.length === 0) return "  (nothing)";
  return changes.map((change) => `  - ${change}`).join("\n");
}

// What the journey saw, in the report: the versions and the file's name.
function observe(testInfo: TestInfo, description: string) {
  testInfo.annotations.push({ type: "observed", description });
}
