import type { Locator, Page, Response, TestInfo } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { getActiveItemId, waitForAgentPageLoad } from "../pages/library.page";
import { expect } from "./fixtures";

const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";

// The typed inputs of backend/blocks/io.py, each with its own widget in the
// Library's run form. Google Drive file is left out: it needs OAuth.
const INPUT_BLOCKS = {
  "short text": "7fcd3bcb-8e1b-4e69-903d-32d3d4a92158",
  "long text": "90a56ffb-7024-4b2b-ab50-e26c5e5ab8ba",
  number: "96dae2bb-97a2-41c2-bd2f-13a3b5a8ea98",
  date: "7e198b09-4994-47db-8b4d-952d98241817",
  time: "2a1c757e-86cf-4c7e-aacf-060dc382e434",
  dropdown: "655d6fdf-a334-421c-b733-520549c07cd1",
  toggle: "cbf36ab5-df4a-43b6-8a7f-f7ed8652116e",
  table: "5603b273-f41e-4020-af7d-fbc9c6a8d928",
  file: "95ead23f-8283-4654-aef3-10c053b74a31",
} as const;

// Runs start INCOMPLETE, then QUEUED, so wait for a status a run ends in.
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

export interface TypedInput {
  name: string;
  kind: keyof typeof INPUT_BLOCKS;
  /** The block's own settings, such as a dropdown's options. */
  settings?: Record<string, unknown>;
}

export interface Agent {
  graphID: string;
  libraryAgentID: string;
}

export interface Run {
  status: string;
  outputs: Record<string, unknown[]>;
}

export interface TimeOfDay {
  hour: string;
  minute: string;
  meridiem: "AM" | "PM";
}

/**
 * Makes a workflow that hands each input straight to an Agent Output of the
 * same name, so a run's outputs are exactly what the workflow got.
 */
export async function createTypedInputsAgent(
  page: Page,
  inputs: TypedInput[],
): Promise<Agent> {
  const graph = {
    name: createUniqueAgentName("Journey typed inputs"),
    description: "Hands each input to an output, for the typed inputs journey",
    is_active: true,
    nodes: inputs.flatMap((input, row) => [
      node(`${input.name}-in`, INPUT_BLOCKS[input.kind], row, 0, {
        name: input.name,
        ...input.settings,
      }),
      node(`${input.name}-out`, AGENT_OUTPUT_BLOCK_ID, row, 1, {
        name: input.name,
      }),
    ]),
    links: inputs.map((input) => ({
      source_id: `${input.name}-in`,
      source_name: "result",
      sink_id: `${input.name}-out`,
      sink_name: "value",
    })),
  };
  const created = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph },
  });
  expect(created.ok(), `creating the workflow: ${created.status()}`).toBe(true);
  const { id: graphID } = (await created.json()) as { id: string };
  const library = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(library.ok(), `reading the Library agent: ${library.status()}`).toBe(
    true,
  );
  const { id: libraryAgentID } = (await library.json()) as { id: string };
  return { graphID, libraryAgentID };
}

function node(
  id: string,
  blockID: string,
  row: number,
  column: number,
  input: Record<string, unknown>,
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x: column * 500, y: row * 200 } },
    input_links: [],
    output_links: [],
  };
}

/** Opens the run form from a new workflow's empty Library page. */
export async function openRunDialog(page: Page, agent: Agent) {
  await page.goto(`/library/agents/${agent.libraryAgentID}`);
  await waitForAgentPageLoad(page);
  // The page settles on its empty state once its run list has loaded.
  const setUp = page.getByRole("button", { name: "Setup your task" });
  await expect(setUp).toBeVisible();
  await setUp.click();
  const dialog = page.getByRole("dialog", { name: "Run Workflow" });
  await expect(dialog).toBeVisible();
  return dialog;
}

/** A text, number or date field, by its input's name. */
export function inputField(dialog: Locator, name: string) {
  return dialog.getByTestId(`agent-input-${name}`);
}

/** A long text field: its textarea is named after the input. */
export function longTextField(dialog: Locator, name: string) {
  return dialog.getByRole("textbox", { name, exact: true });
}

export async function pickOption(
  page: Page,
  combobox: Locator,
  option: string,
) {
  await combobox.click();
  await page.getByRole("option", { name: option, exact: true }).click();
  await expect(combobox, `the field should show ${option}`).toHaveText(option);
}

export async function pickDropdownOption(
  page: Page,
  dialog: Locator,
  name: string,
  option: string,
) {
  await pickOption(
    page,
    dialog.getByRole("combobox", { name, exact: true }),
    option,
  );
}

/** The time field is three selects: hour, minute and AM/PM. */
export async function pickTime(page: Page, dialog: Locator, time: TimeOfDay) {
  const select = (name: string) =>
    dialog.getByRole("combobox", { name, exact: true });
  await pickOption(page, select("Hour"), time.hour);
  await pickOption(page, select("Minute"), time.minute);
  await pickOption(page, select("AM/PM"), time.meridiem);
}

/** Switches a toggle on, and back off again for `false`, as people do. */
export async function setToggle(dialog: Locator, name: string, on: boolean) {
  const toggle = inputField(dialog, name);
  await toggle.click();
  await expect(toggle).toBeChecked();
  if (on) return;
  await toggle.click();
  await expect(toggle).not.toBeChecked();
}

export async function fillTable(
  dialog: Locator,
  rows: Record<string, string>[],
) {
  for (const [index, row] of rows.entries()) {
    await dialog.getByRole("button", { name: "Add Row" }).click();
    for (const [column, value] of Object.entries(row)) {
      await dialog
        .getByPlaceholder(`Enter ${column}`, { exact: true })
        .nth(index)
        .fill(value);
    }
  }
}

export async function uploadFile(
  page: Page,
  dialog: Locator,
  file: { name: string; mimeType: string; buffer: Buffer },
) {
  const uploaded = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.endsWith("/api/files/upload"),
  );
  await dialog.locator('input[type="file"]').setInputFiles(file);
  const response = await uploaded;
  expect(response.ok(), `uploading the file: ${response.status()}`).toBe(true);
  await expect(
    dialog.locator('[data-slot="file-input-file-name"]'),
    "the form should show the uploaded file",
  ).toBeVisible();
}

/**
 * Starts the task the form is filled in for, and leaves the page on its run.
 * Returns the run and the inputs the form sent.
 */
export async function startTask(page: Page, agent: Agent, dialog: Locator) {
  const started = page.waitForResponse(isRunStart(agent.graphID));
  await dialog.getByRole("button", { name: "Start Task", exact: true }).click();
  return openedRun(page, await started);
}

/** Reruns the open run from its view, which then moves to the new run. */
export async function rerunTask(page: Page, agent: Agent) {
  const started = page.waitForResponse(isRunStart(agent.graphID));
  await page.getByRole("button", { name: "Rerun task" }).click();
  return openedRun(page, await started);
}

async function openedRun(page: Page, response: Response) {
  expect(response.ok(), `starting the task: ${response.status()}`).toBe(true);
  const { id: runID } = (await response.json()) as { id: string };
  await expect
    .poll(() => getActiveItemId(page), {
      message: "the run view should open on the new run",
    })
    .toBe(runID);
  const { inputs: sent } = response.request().postDataJSON() as {
    inputs: Record<string, unknown>;
  };
  return { runID, sent };
}

export function isRunStart(graphID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${graphID}/execute/`,
    );
}

export async function waitForRunToEnd(page: Page, agent: Agent, runID: string) {
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

// The run view's Output tab shows each output as an item: its label, and the
// value shown beside it.
const OUTPUT_LABEL = "output-item-label";
const OUTPUT_VALUE = "output-item-value";

/**
 * What the run view's Output tab shows: each output's label, with the text
 * of what is shown for it.
 */
export function readShownOutputs(page: Page) {
  return page
    .getByTestId(OUTPUT_LABEL)
    .filter({ visible: true })
    .evaluateAll(
      (labels, value) =>
        Object.fromEntries(
          labels.map((label) => [
            label.textContent ?? "",
            label.parentElement?.querySelector(`[data-testid="${value}"]`)
              ?.textContent ?? "",
          ]),
        ),
      OUTPUT_VALUE,
    );
}

/** What the run view's Output tab shows for one output. */
export function shownOutput(page: Page, name: string) {
  return page
    .getByTestId(OUTPUT_LABEL)
    .filter({ visible: true, hasText: new RegExp(`^${escapeRegExp(name)}$`) })
    .locator("..")
    .getByTestId(OUTPUT_VALUE);
}

/** Whether `locator` shows within `timeout` ms. */
export function showsWithin(locator: Locator, timeout: number) {
  return locator.waitFor({ timeout }).then(
    () => true,
    () => false,
  );
}

export function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// What the journey saw, in the report: details worth reading next to the
// assertions, such as the inputs the run form sent.
export function observe(testInfo: TestInfo, description: string) {
  testInfo.annotations.push({ type: "observed", description });
}
