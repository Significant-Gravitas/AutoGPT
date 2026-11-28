import type {
  APIRequestContext,
  Locator,
  Page,
  Response,
} from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { markChangelogSeen } from "../utils/changelog";
import {
  assertRunOutputValue,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { backendURL } from "./backend";
import { expect, test as journeyTest } from "./fixtures";
import { watchForUnexpectedErrors } from "./guard";
import { JOURNEY_USER_AGENT } from "./user-agent";

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const OUTPUT_NAME = "report";
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;
// The public page reads a share through the app's proxy.
const PUBLIC_READ = "/api/proxy/api/public/shared/";
// One run's read, by its share token (a UUID), and nothing else under it.
const PUBLIC_RUN_READ = /\/api\/proxy\/api\/public\/shared\/[0-9a-f-]{36}$/;
// All the public read may carry: nothing of the owner's or the graph's.
const PUBLIC_FIELDS = [
  "created_at",
  "graph_description",
  "graph_name",
  "id",
  "outputs",
  "status",
];

interface OwnedRun {
  libraryAgentID: string;
  runID: string;
  ownerID: string;
  // What the owner ran it with: theirs alone.
  note: string;
  // What the run wrote: what a share shows.
  report: string;
}

interface Share {
  token: string;
  link: string;
}

interface Run {
  status: string;
  user_id: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown[]>;
}

const test = journeyTest.extend<{ visitor: Page; backend: APIRequestContext }>({
  // A signed-out visitor: a browser with no session, watched by the same
  // guard as the owner's, so a public page that needs an account fails.
  visitor: async ({ browser }, provide) => {
    const context = await browser.newContext({
      storageState: { cookies: [], origins: [] },
      userAgent: JOURNEY_USER_AGENT,
    });
    await markChangelogSeen(context);
    // A revoked link's run read answers 404; each check asserts when it must.
    const guard = watchForUnexpectedErrors(context, [
      { status: 404, method: "GET", url: PUBLIC_RUN_READ },
    ]);
    await provide(await context.newPage());
    const problems = await guard.collect();
    await context.close();
    expect(
      problems,
      `the visitor hit ${problems.length} unexpected error(s):\n\n` +
        problems.join("\n\n"),
    ).toEqual([]);
  },
  // The backend called directly with no cookies, as anyone with a link can.
  backend: async ({ playwright }, provide) => {
    const backend = await playwright.request.newContext({
      storageState: { cookies: [], origins: [] },
    });
    await provide(backend);
    await backend.dispose();
  },
});

test.beforeEach(async ({ context }) => {
  // The owner hands each link on with the dialog's copy button.
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
});

test.describe("run sharing journey", () => {
  test("a signed-out visitor sees a shared run until the owner unshares it, and sharing again gives a new link", async ({
    page,
    journeyUser,
    visitor,
    backend,
  }) => {
    const run = await createFinishedRun(page);
    const dialog = await openShareDialog(page, run);

    const first = await enableSharing(page, dialog, run);
    expect(
      await publicStatus(backend, first.token),
      "the backend should serve a live link",
    ).toBe(200);
    await expectVisitorSeesReport(visitor, first, run, journeyUser.email);

    await stopSharing(page, dialog, run);
    await expectVisitorSeesNotFound(visitor, first, run);
    expect(
      await publicStatus(backend, first.token),
      "the backend should refuse an unshared link",
    ).toBe(404);

    const second = await enableSharing(page, dialog, run);
    expect(second.token, "sharing again should give a new token").not.toBe(
      first.token,
    );
    expect(
      await publicStatus(backend, first.token),
      "sharing again should leave the old link dead",
    ).toBe(404);
    expect(
      await publicStatus(backend, second.token),
      "the backend should serve the new link",
    ).toBe(200);
    await expectVisitorSeesReport(visitor, second, run, journeyUser.email);
  });

  test("deleting a shared run takes its link down for the visitor", async ({
    page,
    journeyUser,
    visitor,
    backend,
  }) => {
    const run = await createFinishedRun(page);
    const dialog = await openShareDialog(page, run);
    const share = await enableSharing(page, dialog, run);
    expect(
      await publicStatus(backend, share.token),
      "the backend should serve a live link",
    ).toBe(200);
    await expectVisitorSeesReport(visitor, share, run, journeyUser.email);

    await dialog.getByRole("button", { name: "Close" }).click();
    await expect(dialog).toBeHidden();
    await deleteRun(page, run);

    await expectVisitorSeesNotFound(visitor, share, run);
    expect(
      await publicStatus(backend, share.token),
      "the backend should refuse a deleted run's link",
    ).toBe(404);
  });
});

// AgentInput "note" starts a Store Value holding the report, which an
// AgentOutput writes. Only the report should ever reach a visitor.
async function createFinishedRun(page: Page): Promise<OwnedRun> {
  const stamp = Date.now();
  const note = `Private note ${stamp}`;
  const report = `Journey report ${stamp}`;
  const created = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name: createUniqueAgentName("Journey run sharing"),
        description: "Writes a report, for the run sharing journey",
        is_active: true,
        nodes: [
          node("note", AGENT_INPUT_BLOCK_ID, { name: "note" }, [0, 0]),
          node("store", STORE_VALUE_BLOCK_ID, { data: report }, [400, 0]),
          node(
            "report",
            AGENT_OUTPUT_BLOCK_ID,
            { name: OUTPUT_NAME },
            [800, 0],
          ),
        ],
        links: [
          link("note", "result", "store", "input"),
          link("store", "output", "report", "value"),
        ],
      },
    },
  });
  expect(created.ok(), `creating the agent: ${created.status()}`).toBe(true);
  const graph = (await created.json()) as { id: string; version: number };

  const started = await page.request.post(
    `/api/proxy/api/graphs/${graph.id}/execute/${graph.version}`,
    { data: { inputs: { note }, credentials_inputs: {} } },
  );
  expect(started.ok(), `starting the run: ${started.status()}`).toBe(true);
  const runID = ((await started.json()) as { id: string }).id;

  // Runs start INCOMPLETE, then QUEUED, so wait for one that has ended.
  await expect
    .poll(async () => (await readRun(page, graph.id, runID)).status, {
      message: "the run should end",
      timeout: 30_000,
    })
    .toMatch(RUN_ENDED);
  const run = await readRun(page, graph.id, runID);
  expect(run.status).toBe("COMPLETED");
  expect(run.outputs, "the run should write the report").toEqual({
    [OUTPUT_NAME]: [report],
  });
  expect(run.inputs, "the owner's run should keep the note").toMatchObject({
    note,
  });

  return {
    libraryAgentID: await readLibraryAgentID(page, graph.id),
    runID,
    ownerID: run.user_id,
    note,
    report,
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

async function readRun(page: Page, graphID: string, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as Run;
}

async function readLibraryAgentID(page: Page, graphID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graphID}`,
  );
  expect(response.ok(), `reading the Library agent: ${response.status()}`).toBe(
    true,
  );
  return ((await response.json()) as { id: string }).id;
}

// The owner opens the finished run in the Library, then its Share control,
// which the run view shows once the run has loaded.
async function openShareDialog(page: Page, run: OwnedRun) {
  await page.goto(
    `/library/agents/${run.libraryAgentID}?activeTab=runs&activeItem=${run.runID}`,
  );
  await waitForAgentPageLoad(page);
  await assertRunOutputValue(page, OUTPUT_NAME, run.report);
  await page.getByRole("button", { name: "Share results" }).click();
  const dialog = page.getByRole("dialog", { name: "Share Workflow Run" });
  await expect(
    dialog,
    "the Share control should open its dialog",
  ).toBeVisible();
  return dialog;
}

// Shares the run and copies the link the dialog gives, as the owner would to
// send it on.
async function enableSharing(
  page: Page,
  dialog: Locator,
  run: OwnedRun,
): Promise<Share> {
  const shared = page.waitForResponse(isShareChange(run.runID, "POST"));
  await dialog.getByRole("button", { name: "Enable Sharing" }).click();
  const response = await shared;
  expect(response.ok(), `sharing the run: ${response.status()}`).toBe(true);
  const { share_token: token } = (await response.json()) as {
    share_token: string;
  };
  const link = new URL(`/share/${token}`, page.url()).href;

  const field = dialog.getByRole("textbox", { name: "Share URL" });
  await expect(
    field,
    "the dialog should show the new share's link",
  ).toHaveValue(link);
  await copyButton(field).click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()), {
      message: "the copy button should put the link on the clipboard",
    })
    .toBe(link);
  return { token, link };
}

// The copy button has no name of its own: it is the button beside the link.
function copyButton(field: Locator) {
  return field.locator("xpath=ancestor::div[.//button][1]").getByRole("button");
}

async function stopSharing(page: Page, dialog: Locator, run: OwnedRun) {
  const stopped = page.waitForResponse(isShareChange(run.runID, "DELETE"));
  await dialog.getByRole("button", { name: "Stop Sharing" }).click();
  const response = await stopped;
  expect(response.ok(), `unsharing the run: ${response.status()}`).toBe(true);
  await expect(
    dialog.getByRole("button", { name: "Enable Sharing" }),
    "the dialog should offer to share again",
  ).toBeVisible();
}

// Each run in the sidebar has a "More actions" menu too; the run view's own
// sits beside its Share control, and clears the selection once it deletes.
async function deleteRun(page: Page, run: OwnedRun) {
  await page
    .getByRole("button", { name: "Share results" })
    .locator("xpath=ancestor::div[.//button[@aria-label='More actions']][1]")
    .getByRole("button", { name: "More actions" })
    .click();
  await page.getByRole("menuitem", { name: "Delete this task" }).click();
  const confirm = page.getByRole("dialog", { name: "Delete task" });
  const deleted = page.waitForResponse(isRunDelete(run.runID));
  await confirm
    .getByRole("button", { name: "Delete Task", exact: true })
    .click();
  const response = await deleted;
  expect(response.ok(), `deleting the run: ${response.status()}`).toBe(true);
  await expect(
    page.getByTestId(`library-run-item-${run.runID}`),
    "the Library should drop the deleted run",
  ).toHaveCount(0);
}

// A live link shows a signed-out visitor the run's report, and nothing that
// is the owner's alone: not the note they ran it with, nor who they are.
async function expectVisitorSeesReport(
  visitor: Page,
  share: Share,
  run: OwnedRun,
  ownerEmail: string,
) {
  const read = visitor.waitForResponse(isPublicRead(share.token));
  await visitor.goto(share.link);
  const response = await read;
  expect(response.status(), "the public page should read the share").toBe(200);
  const body = await response.text();
  const shared = JSON.parse(body) as Record<string, unknown>;
  expect(
    Object.keys(shared).sort(),
    "the share should carry only its public fields",
  ).toEqual(PUBLIC_FIELDS);
  expect(shared.outputs, "the share should carry the run's report").toEqual({
    [OUTPUT_NAME]: [run.report],
  });

  await expect(visitor, "the link should open without signing in").toHaveURL(
    share.link,
  );
  await assertRunOutputValue(visitor, OUTPUT_NAME, run.report);
  await expect(
    visitor.getByRole("link", { name: "Sign up" }),
    "the visitor should be signed out",
  ).toBeVisible();

  const html = await visitor.content();
  const ownerOnly = {
    note: run.note,
    "user id": run.ownerID,
    email: ownerEmail,
  };
  for (const [what, value] of Object.entries(ownerOnly)) {
    expect(
      body,
      `the share should not carry the owner's ${what}`,
    ).not.toContain(value);
    expect(html, `the page should not show the owner's ${what}`).not.toContain(
      value,
    );
  }
}

// The page shows its not-found state once its read of the share answers
// 404. It retries a failed read three times first, about 7 s.
async function expectVisitorSeesNotFound(
  visitor: Page,
  share: Share,
  run: OwnedRun,
) {
  await expect(visitor).toHaveURL(share.link);
  const read = visitor.waitForResponse(isPublicRead(share.token));
  await visitor.reload();
  expect(
    (await read).status(),
    "the public page's read of a revoked link",
  ).toBe(404);
  await expect(
    visitor.getByRole("heading", { name: "Share Link Not Found" }),
    "the visitor should see the page's not-found state",
  ).toBeVisible({ timeout: 30_000 });
  await expect(
    visitor.getByText(run.report),
    "the visitor should no longer see the report",
  ).toHaveCount(0);
}

// What the backend itself answers for a link, asked with no session.
async function publicStatus(backend: APIRequestContext, token: string) {
  return (
    await backend.get(`${backendURL()}/api/public/shared/${token}`)
  ).status();
}

function isShareChange(runID: string, method: "POST" | "DELETE") {
  return (response: Response) =>
    response.request().method() === method &&
    new URL(response.url()).pathname.endsWith(`/executions/${runID}/share`);
}

function isRunDelete(runID: string) {
  return (response: Response) =>
    response.request().method() === "DELETE" &&
    new URL(response.url()).pathname.endsWith(`/api/executions/${runID}`);
}

function isPublicRead(token: string) {
  return (response: Response) =>
    response.request().method() === "GET" &&
    new URL(response.url()).pathname === `${PUBLIC_READ}${token}`;
}
