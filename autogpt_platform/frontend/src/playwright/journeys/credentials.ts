import { randomBytes } from "crypto";
import {
  expect,
  type Locator,
  type Page,
  type Response,
  type TestInfo,
} from "@playwright/test";
import { authHeaders, signIn, type JourneyUser } from "./journey-user";

// The credentials journey's steps. Its keys are for Firecrawl, whose scrape
// block takes an API key and which has no platform key to fall back on, so
// the key is always the user's own. No key is real, and the step that would
// send one never runs: the journey's workflow only reaches it when its note
// is a word nobody types (see `scrapingGraph`), so a run checks the key is
// there and usable without calling the provider.

export const PROVIDER = "firecrawl";
export const PROVIDER_NAME = "Firecrawl";

const BLOCK = {
  input: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  output: "363ae599-353e-4804-937e-b2ee3cef3da4",
  condition: "715696a0-e1da-45c8-b209-c2fa9c3b0be6",
  scrape: "ac444320-cf5e-4697-b586-2604c17a3e75",
} as const;
const NEVER_TYPED = `never-typed-${randomBytes(6).toString("hex")}`;
// Runs start INCOMPLETE, then QUEUED, so wait for a status a run ends in.
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

export interface Credential {
  id: string;
  provider: string;
  type: string;
  title: string;
  service_name?: string | null;
}

export interface Workflow {
  name: string;
  graphID: string;
  graphVersion: number;
  libraryAgentID: string;
  /** The key in the workflow's credentials inputs that the scrape step reads. */
  credentialsInput: string;
}

export interface Run {
  id: string;
  status: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown[]>;
  credential_inputs: Record<string, CredentialRef> | null;
}

export interface CredentialRef {
  id: string;
  provider: string;
  type: string;
  title?: string | null;
}

export interface Template {
  id: string;
  name: string;
  credentials: Record<string, CredentialRef>;
}

export interface Schedule {
  id: string;
  name: string;
  cron: string;
  input_credentials: Record<string, CredentialRef>;
}

/** A word no other journey's names contain. */
export function uniqueStamp() {
  return `${Date.now()}${randomBytes(2).toString("hex")}`;
}

/** A key that looks like one and works nowhere. */
export function fakeAPIKey() {
  return `journey-not-a-key-${randomBytes(12).toString("hex")}`;
}

export function credentialRef(credential: Credential): CredentialRef {
  return {
    id: credential.id,
    provider: credential.provider,
    type: credential.type,
    title: credential.title,
  };
}

// ── Set-up through the API ─────────────────────────────────────────────

export async function createAPIKey(page: Page, title: string) {
  const response = await page.request.post(
    `/api/proxy/api/integrations/${PROVIDER}/credentials`,
    {
      data: {
        type: "api_key",
        provider: PROVIDER,
        title,
        api_key: fakeAPIKey(),
      },
    },
  );
  expect(response.ok(), `saving the API key: ${response.status()}`).toBe(true);
  return (await response.json()) as Credential;
}

export async function listCredentials(page: Page) {
  const response = await page.request.get(
    "/api/proxy/api/integrations/credentials",
  );
  expect(response.ok(), `listing credentials: ${response.status()}`).toBe(true);
  return {
    text: await response.text(),
    credentials: (await response.json()) as Credential[],
  };
}

/** Saves the journey's workflow, its scrape step on `credential` if given. */
export async function createWorkflow(
  page: Page,
  name: string,
  credential?: Credential,
): Promise<Workflow> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph: scrapingGraph(name, credential) },
  });
  expect(response.ok(), `saving ${name}: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  const agent = await readLibraryAgent(page, graph.id);
  const inputs = Object.keys(agent.credentials_input_schema.properties);
  expect(
    inputs,
    "the workflow should ask for one Firecrawl key, for its scrape step",
  ).toHaveLength(1);
  return {
    name,
    graphID: graph.id,
    graphVersion: graph.version,
    libraryAgentID: agent.id,
    credentialsInput: inputs[0],
  };
}

async function readLibraryAgent(page: Page, graphID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(response.ok(), `reading the Library entry: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as {
    id: string;
    graph_version: number;
    credentials_input_schema: { properties: Record<string, unknown> };
  };
}

// note → reply, and note → page only when the note is NEVER_TYPED, through a
// scrape step that reads the key.
function scrapingGraph(name: string, credential?: Credential) {
  return {
    name,
    description: "Made by the credentials journey",
    is_active: true,
    nodes: [
      graphNode("note", BLOCK.input, { name: "note", title: "Note" }, 0, 0),
      graphNode("reply", BLOCK.output, { name: "reply", title: "Reply" }, 800),
      graphNode(
        "gate",
        BLOCK.condition,
        { operator: "==", value2: NEVER_TYPED },
        400,
        320,
      ),
      graphNode(
        "scrape",
        BLOCK.scrape,
        credential ? { credentials: credentialRef(credential) } : {},
        800,
        320,
      ),
      graphNode(
        "page",
        BLOCK.output,
        { name: "page", title: "Page" },
        1200,
        320,
      ),
    ],
    links: [
      graphLink("note", "result", "reply", "value"),
      graphLink("note", "result", "gate", "value1"),
      graphLink("gate", "yes_output", "scrape", "url"),
      graphLink("scrape", "markdown", "page", "value"),
    ],
  };
}

function graphNode(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  x: number,
  y = 0,
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

function graphLink(
  source: string,
  output: string,
  sink: string,
  input: string,
) {
  return {
    source_id: source,
    source_name: output,
    sink_id: sink,
    sink_name: input,
  };
}

/** The saved graph's scrape step: its id and the key its field holds. */
export async function readScrapeStep(page: Page, workflow: Workflow) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graphID}`,
  );
  expect(response.ok(), `reading the workflow: ${response.status()}`).toBe(
    true,
  );
  const graph = (await response.json()) as {
    version: number;
    nodes: {
      id: string;
      block_id: string;
      input_default: Record<string, unknown>;
    }[];
  };
  const step = graph.nodes.find((node) => node.block_id === BLOCK.scrape);
  expect(step, "the workflow should have its scrape step").toBeTruthy();
  return {
    version: graph.version,
    nodeID: step!.id,
    credentials: (step!.input_default.credentials ?? null) as {
      id?: string;
    } | null,
  };
}

/** Starts a run through the API, as an integration would. */
export async function startRun(
  page: Page,
  workflow: Workflow,
  note: string,
  credential?: CredentialRef,
) {
  return page.request.post(
    `/api/proxy/api/graphs/${workflow.graphID}/execute/${workflow.graphVersion}`,
    {
      data: {
        inputs: { note },
        credentials_inputs: credential
          ? { [workflow.credentialsInput]: credential }
          : {},
      },
    },
  );
}

export async function waitForRunToEnd(
  page: Page,
  workflow: Workflow,
  runID: string,
) {
  await expect
    .poll(async () => (await readRun(page, workflow, runID)).status, {
      message: "the run should end",
      timeout: 30_000,
    })
    .toMatch(RUN_ENDED);
  return readRun(page, workflow, runID);
}

async function readRun(page: Page, workflow: Workflow, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graphID}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as Run;
}

export async function listRunIDs(page: Page, workflow: Workflow) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graphID}/executions`,
  );
  expect(response.ok(), `listing the runs: ${response.status()}`).toBe(true);
  const { executions } = (await response.json()) as {
    executions: { id: string }[];
  };
  return executions.map((run) => run.id);
}

export async function createTemplate(page: Page, runID: string, name: string) {
  const response = await page.request.post("/api/proxy/api/library/presets", {
    data: {
      name,
      description: "Made by the credentials journey",
      graph_execution_id: runID,
    },
  });
  expect(response.ok(), `saving the template: ${response.status()}`).toBe(true);
  return (await response.json()) as Template;
}

export async function readTemplate(page: Page, templateID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/presets/${templateID}`,
  );
  expect(response.ok(), `reading the template: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as Template;
}

export async function createSchedule(
  page: Page,
  workflow: Workflow,
  schedule: { name: string; cron: string; credential: CredentialRef },
) {
  const response = await page.request.post(
    `/api/proxy/api/graphs/${workflow.graphID}/schedules`,
    {
      data: {
        name: schedule.name,
        cron: schedule.cron,
        inputs: { note: `Scheduled ${schedule.name}` },
        credentials: { [workflow.credentialsInput]: schedule.credential },
        timezone: "UTC",
      },
    },
  );
  expect(
    response.ok(),
    `creating the schedule: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as Schedule;
}

export async function listSchedules(page: Page, workflow: Workflow) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${workflow.graphID}/schedules`,
  );
  expect(response.ok(), `listing the schedules: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as Schedule[];
}

/** A schedule left behind fires into the next journey's stack, so a delete
 * that fails is reported, softly, after the test's own steps. */
export async function deleteSchedule(page: Page, scheduleID: string) {
  const response = await page.request.delete(
    `/api/proxy/api/schedules/${scheduleID}`,
  );
  expect
    .soft(
      response.ok() || response.status() === 404,
      `deleting schedule ${scheduleID}: ${response.status()}`,
    )
    .toBe(true);
}

/** Removes a key a test made, so a reused account doesn't collect them. */
export async function deleteCredential(page: Page, credential: Credential) {
  const response = await page.request.delete(
    `/api/proxy/api/integrations/${credential.provider}/credentials/${credential.id}?force=true`,
  );
  expect
    .soft(
      response.ok() || response.status() === 404,
      `deleting key ${credential.id}: ${response.status()}`,
    )
    .toBe(true);
}

// ── Settings → Integrations ────────────────────────────────────────────

export async function openIntegrations(page: Page) {
  await page.goto("/settings/integrations");
  await expect(
    page.getByRole("heading", { name: "Integrations", exact: true }),
  ).toBeVisible();
  await expect(
    integrationsSearch(page),
    "the list should finish loading",
  ).toBeEnabled();
}

function integrationsSearch(page: Page) {
  return page.getByRole("textbox", { name: "Search integrations" });
}

/** Adds an API key the way the page offers: Connect Service, then the form. */
export async function addKeyInIntegrations(
  page: Page,
  key: { title: string; apiKey: string },
) {
  await page
    .getByRole("button", { name: "Connect Service" })
    .filter({ visible: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "Connect a service" });
  await dialog
    .getByRole("textbox", { name: "Search services" })
    .fill(PROVIDER_NAME);
  // The list filters on a debounce and re-renders, so a click can land on a
  // row that's being replaced; click until the key form opens.
  const nameField = dialog.getByRole("textbox", { name: "Name", exact: true });
  await expect(async () => {
    // Once the form has opened the service list is gone, so a retry only
    // waits for the form.
    const serviceButton = dialog.getByRole("button", {
      name: new RegExp(PROVIDER_NAME),
    });
    if (await serviceButton.isVisible()) {
      await serviceButton.click({ timeout: 2_000 });
    }
    await expect(nameField).toBeVisible({ timeout: 2_000 });
  }).toPass();
  await nameField.fill(key.title);
  await dialog.getByLabel("API key", { exact: true }).fill(key.apiKey);
  const created = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.endsWith(
        `/api/integrations/${PROVIDER}/credentials`,
      ),
  );
  await dialog.getByRole("button", { name: "Save API key" }).click();
  const response = await created;
  expect(response.ok(), `saving the key: ${response.status()}`).toBe(true);
  expect(
    await response.text(),
    "saving a key should not answer with the key",
  ).not.toContain(key.apiKey);
  await expect(
    dialog,
    "the dialog should close once the key is saved",
  ).toBeHidden();
  return (await response.json()) as Credential;
}

/** The provider's group in the list: its heading names the service. */
export function serviceGroup(page: Page, serviceName: string) {
  const escaped = serviceName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return page.getByRole("region", { name: new RegExp(escaped) });
}

export function keyCheckbox(scope: Page | Locator, title: string) {
  return scope.getByRole("checkbox", { name: `Select ${title}`, exact: true });
}

export async function expectListedInIntegrations(
  page: Page,
  credential: Credential,
  serviceName: string,
) {
  const group = serviceGroup(page, serviceName);
  await expect(
    keyCheckbox(group, credential.title),
    "the key should be listed under its service",
  ).toBeVisible();
  const row = keyCheckbox(group, credential.title).locator("xpath=../..");
  await expect(row.getByText("API Key", { exact: true })).toBeVisible();
  await expect(row.getByText(/^API key configured$/i)).toBeVisible();
}

export async function removeKeyInIntegrations(page: Page, title: string) {
  await page
    .getByRole("button", { name: `Delete ${title}`, exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: `Remove ${title}?` });
  await expect(dialog).toBeVisible();
  const warning = (await dialog.innerText()).replace(/\s+/g, " ").trim();
  const deleted = page.waitForResponse(
    (response) =>
      response.request().method() === "DELETE" &&
      new URL(response.url()).pathname.includes(
        `/api/integrations/${PROVIDER}/credentials/`,
      ),
  );
  await dialog.getByRole("button", { name: "Remove", exact: true }).click();
  const response = await deleted;
  expect(response.ok(), `removing the key: ${response.status()}`).toBe(true);
  return {
    warning,
    answer: (await response.json()) as Record<string, unknown>,
  };
}

// ── The Library ────────────────────────────────────────────────────────

export function runDialog(page: Page) {
  return page.getByRole("dialog", { name: "Run Workflow" });
}

export function isRunStart(workflow: Workflow) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${workflow.graphID}/execute/`,
    );
}

export function isTemplateStart(templateID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith(
      `/api/library/presets/${templateID}/execute`,
    );
}

// ── Sessions ───────────────────────────────────────────────────────────

/** Ends the session and starts a new one with no page of the app open. */
export async function signOutAndBackIn(page: Page, user: JourneyUser) {
  await page.goto("about:blank");
  const signedOut = await page.request.post("/api/auth/sign-out", {
    headers: authHeaders(),
    data: {},
  });
  expect(signedOut.ok(), `signing out: ${signedOut.status()}`).toBe(true);
  const refused = await page.request.get(
    "/api/proxy/api/integrations/credentials",
  );
  expect(refused.status(), "signed out, the keys should be out of reach").toBe(
    401,
  );
  await signIn(page, user);
}

/** What the journey saw, in the report, next to what it asserted. */
export function observe(testInfo: TestInfo, description: string) {
  testInfo.annotations.push({ type: "observed", description });
}
