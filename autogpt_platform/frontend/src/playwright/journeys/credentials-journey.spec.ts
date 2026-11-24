import type { Browser, Page } from "@playwright/test";
import { BuildPage, createUniqueAgentName } from "../pages/build.page";
import { waitForAgentPageLoad } from "../pages/library.page";
import { markChangelogSeen } from "../utils/changelog";
import {
  addKeyInIntegrations,
  createAPIKey,
  createSchedule,
  createTemplate,
  createWorkflow,
  credentialRef,
  deleteCredential,
  deleteSchedule,
  expectListedInIntegrations,
  fakeAPIKey,
  isRunStart,
  isTemplateStart,
  keyCheckbox,
  listCredentials,
  listRunIDs,
  listSchedules,
  observe,
  openIntegrations,
  PROVIDER,
  PROVIDER_NAME,
  readScrapeStep,
  readTemplate,
  removeKeyInIntegrations,
  runDialog,
  signOutAndBackIn,
  startRun,
  uniqueStamp,
  waitForRunToEnd,
  type Credential,
  type Workflow,
} from "./credentials";
import { expect, serveHostedFlags, test as journeyTest } from "./fixtures";
import { watchForUnexpectedErrors } from "./guard";
import { type JourneyUser, provisionNewJourneyUser } from "./journey-user";
import { JOURNEY_USER_AGENT } from "./user-agent";

interface Person {
  page: Page;
  user: JourneyUser;
}

const test = journeyTest.extend<{ someoneElse: Person }>({
  // Another user, in a browser of their own, who must never reach the
  // journey user's keys.
  someoneElse: async ({ browser, baseURL, flagOverrides }, provide) => {
    await asSomeoneElse(browser, { baseURL, flagOverrides }, provide);
  },
});

// A template started after its key was removed is refused by the server.
const TEMPLATE_START = /\/api\/proxy\/api\/library\/presets\/[^/]+\/execute$/;

test.describe("credentials journey", () => {
  test("an API key added in Settings → Integrations is listed under its service by name, stays through a reload and a new session, and nobody else can see, read, delete or run with it", async ({
    page,
    journeyUser,
    someoneElse,
  }) => {
    const title = `Journey key ${uniqueStamp()}`;
    const apiKey = fakeAPIKey();

    await openIntegrations(page);
    const credential = await addKeyInIntegrations(page, { title, apiKey });
    try {
      expect(credential).toMatchObject({
        provider: PROVIDER,
        type: "api_key",
        title,
      });
      const service = credential.service_name ?? PROVIDER_NAME;
      await expectListedInIntegrations(page, credential, service);
      expect(
        await page.content(),
        "the page should never show the key itself",
      ).not.toContain(apiKey);
      const listed = await listCredentials(page);
      expect(
        listed.credentials.find((saved) => saved.id === credential.id),
      ).toMatchObject({ provider: PROVIDER, type: "api_key", title });
      expect(listed.text, "the API should never return the key").not.toContain(
        apiKey,
      );

      await page.reload();
      await expectListedInIntegrations(page, credential, service);

      await signOutAndBackIn(page, journeyUser);
      await openIntegrations(page);
      await expectListedInIntegrations(page, credential, service);

      // Someone else, in their own browser, through the page and the API.
      const other = someoneElse.page;
      await openIntegrations(other);
      await expect(keyCheckbox(other, title)).toHaveCount(0);
      const theirs = await listCredentials(other);
      expect(theirs.credentials.map((saved) => saved.id)).not.toContain(
        credential.id,
      );
      expect(theirs.text).not.toContain(title);
      const keyURL = `/api/proxy/api/integrations/${PROVIDER}/credentials/${credential.id}`;
      expect(
        (await other.request.get(keyURL)).status(),
        "another user should not be able to read the key",
      ).toBe(404);
      expect(
        (await other.request.delete(keyURL)).status(),
        "another user should not be able to delete the key",
      ).toBe(404);
      const theirWorkflow = await createWorkflow(
        other,
        createUniqueAgentName("Journey borrowed key"),
      );
      const borrowed = await startRun(
        other,
        theirWorkflow,
        "Borrowing a key",
        credentialRef(credential),
      );
      expect(
        borrowed.status(),
        "a run on another user's key should be refused",
      ).toBe(400);
      expect(await borrowed.text()).toContain(
        `Unknown credentials #${credential.id}`,
      );
      expect(await listRunIDs(other, theirWorkflow)).toEqual([]);

      expect(
        (await listCredentials(page)).credentials.map((saved) => saved.id),
        "the key should still be its owner's",
      ).toContain(credential.id);
    } finally {
      await deleteCredential(page, credential);
    }
  });

  test("the key is offered in the builder's credentials field, the run dialog, a template and a schedule, and each keeps it once saved", async ({
    page,
    journeyUser: _user,
  }) => {
    // Five surfaces in one test: about 47 s on CI, so room for a slow runner.
    test.setTimeout(180_000);
    const stamp = uniqueStamp();
    const credential = await createAPIKey(page, `Journey key ${stamp}`);
    let otherKey: Credential | undefined;
    try {
      const workflow = await createWorkflow(
        page,
        createUniqueAgentName("Journey scraping"),
      );
      const before = await readScrapeStep(page, workflow);
      expect(before.credentials, "the step starts with no key").toBeNull();

      // The builder: the field offers the user's only Firecrawl key.
      await page.goto(`/build?flowID=${workflow.graphID}`);
      await new BuildPage(page).closeTutorial();
      const step = page.locator(`[data-id="custom-node-${before.nodeID}"]`);
      await expect(step).toBeVisible({ timeout: 20_000 });
      await expect(
        step.getByText(credential.title, { exact: true }),
      ).toBeVisible();
      await step.getByText(credential.title, { exact: true }).click();
      const saved = await saveInBuilder(page, workflow);
      expect(
        saved.credentials,
        "saving should keep the picked key",
      ).toMatchObject({ id: credential.id, provider: PROVIDER });
      await page.reload();
      await expect(
        page
          .locator(`[data-id="custom-node-${saved.nodeID}"]`)
          .getByText(credential.title, { exact: true }),
        "the builder should still show the key after a reload",
      ).toBeVisible({ timeout: 20_000 });

      // The run dialog: with a second key saved, only the step's choice tells
      // the two apart, so it starts on the key picked on the step, and the
      // run uses that key.
      otherKey = await createAPIKey(page, `Journey other key ${stamp}`);
      await page.goto(`/library/agents/${workflow.libraryAgentID}`);
      await waitForAgentPageLoad(page);
      const setUp = page.getByRole("button", { name: "Setup your task" });
      await expect(setUp).toBeVisible();
      await setUp.click();
      const dialog = runDialog(page);
      await expect(dialog.getByText("Task Credentials")).toBeVisible();
      await expect(
        dialog.getByText(credential.title, { exact: true }),
        "the run dialog should start on the key picked on the step",
      ).toBeVisible();
      const note = `Scraping note ${stamp}`;
      await dialog.getByTestId("agent-input-note").fill(note);
      const started = page.waitForResponse(isRunStart(workflow));
      await dialog
        .getByRole("button", { name: "Start Task", exact: true })
        .click();
      const startedRun = await started;
      expect(startedRun.ok(), `starting the run: ${startedRun.status()}`).toBe(
        true,
      );
      expect(
        startedRun.request().postDataJSON().credentials_inputs,
        "the run dialog should send the key it offered",
      ).toEqual({
        [workflow.credentialsInput]: expect.objectContaining({
          id: credential.id,
        }),
      });
      const runID = ((await startedRun.json()) as { id: string }).id;
      const run = await waitForRunToEnd(page, workflow, runID);
      expect(run.status).toBe("COMPLETED");
      expect(run.outputs, "the scrape step never runs").toEqual({
        reply: [note],
      });
      expect(run.credential_inputs?.[workflow.credentialsInput]).toMatchObject({
        id: credential.id,
      });

      // A template saved from the run keeps the key, and shows it.
      const template = await saveRunAsTemplate(
        page,
        runID,
        `Template ${stamp}`,
      );
      expect(template.credentials).toEqual({
        [workflow.credentialsInput]: expect.objectContaining({
          id: credential.id,
        }),
      });
      await openTemplate(page, workflow, template.id);
      await expect(
        page.getByText(credential.title, { exact: true }),
        "the template should show its key",
      ).toBeVisible();
      await page.reload();
      await waitForAgentPageLoad(page);
      await expect(
        page.getByText(credential.title, { exact: true }),
      ).toBeVisible();
      expect(
        (await readTemplate(page, template.id)).credentials[
          workflow.credentialsInput
        ],
      ).toMatchObject({ id: credential.id });

      // A schedule made from the run dialog keeps the key.
      await page.goto(`/library/agents/${workflow.libraryAgentID}`);
      await waitForAgentPageLoad(page);
      await page.getByRole("button", { name: "New workflow task" }).click();
      await runDialog(page).getByTestId("agent-input-note").fill(note);
      await expect(
        runDialog(page).getByText(credential.title, { exact: true }),
      ).toBeVisible();
      await page.getByRole("button", { name: "Schedule Task" }).click();
      const scheduleDialog = page.getByRole("dialog", { name: "Schedule Run" });
      await scheduleDialog
        .getByLabel("Schedule Name")
        .fill(`Schedule ${stamp}`);
      const scheduled = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname.endsWith(
            `/api/graphs/${workflow.graphID}/schedules`,
          ),
      );
      await scheduleDialog.getByRole("button", { name: "Schedule" }).click();
      const scheduleResponse = await scheduled;
      expect(
        scheduleResponse.ok(),
        `creating the schedule: ${scheduleResponse.status()}`,
      ).toBe(true);
      const schedule = (await scheduleResponse.json()) as { id: string };
      try {
        const [listedSchedule] = await listSchedules(page, workflow);
        expect(listedSchedule?.input_credentials).toEqual({
          [workflow.credentialsInput]: expect.objectContaining({
            id: credential.id,
          }),
        });
      } finally {
        await deleteSchedule(page, schedule.id);
      }
    } finally {
      if (otherKey) await deleteCredential(page, otherKey);
      await deleteCredential(page, credential);
    }
  });
});

test.describe("credentials journey: removing a key in use", () => {
  test.use({
    expectedResponses: [{ status: 400, method: "POST", url: TEMPLATE_START }],
  });

  test("removing a key that a workflow, a template and a schedule use warns before it goes, and the run dialog, the template and the builder each say it is gone", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    test.setTimeout(120_000);
    const stamp = uniqueStamp();
    const credential = await createAPIKey(page, `Journey key ${stamp}`);
    // The test removes the key itself; this covers a failure before then.
    try {
      const workflow = await createWorkflow(
        page,
        createUniqueAgentName("Journey scraping"),
        credential,
      );
      const firstRunID = await startedRunID(
        await startRun(
          page,
          workflow,
          `First ${stamp}`,
          credentialRef(credential),
        ),
      );
      expect((await waitForRunToEnd(page, workflow, firstRunID)).status).toBe(
        "COMPLETED",
      );
      const template = await createTemplate(
        page,
        firstRunID,
        `Template ${stamp}`,
      );
      // Far off, so it never fires during the test.
      const schedule = await createSchedule(page, workflow, {
        name: `Schedule ${stamp}`,
        cron: "0 3 29 2 *",
        credential: credentialRef(credential),
      });

      try {
        await openIntegrations(page);
        const service = credential.service_name ?? PROVIDER_NAME;
        await expectListedInIntegrations(page, credential, service);
        const { warning, answer } = await removeKeyInIntegrations(
          page,
          credential.title,
        );
        observe(
          testInfo,
          `removing a key a workflow, a template and a schedule use, the ` +
            `dialog says: "${warning}"; the server answered ${JSON.stringify(answer)}`,
        );
        expect(warning).toContain(
          "Experts and workflows using this credential will lose access immediately.",
        );
        await expect(keyCheckbox(page, credential.title)).toHaveCount(0);
        expect(
          (await listCredentials(page)).credentials.map((saved) => saved.id),
        ).not.toContain(credential.id);
        observe(
          testInfo,
          `after the key was removed, the workflow's schedules are: ` +
            JSON.stringify(
              (await listSchedules(page, workflow)).map((listed) => ({
                id: listed.id,
                key: listed.input_credentials?.[workflow.credentialsInput]?.id,
              })),
            ),
        );

        // The run dialog: nothing to pick, so the task can't start.
        await page.goto(`/library/agents/${workflow.libraryAgentID}`);
        await waitForAgentPageLoad(page);
        await page.getByRole("button", { name: "New workflow task" }).click();
        const dialog = runDialog(page);
        await dialog.getByTestId("agent-input-note").fill(`After ${stamp}`);
        await expect(
          dialog.getByRole("button", { name: "Add API key" }),
          "the run dialog should offer to add a key",
        ).toBeVisible();
        await expect(
          dialog.getByText(credential.title, { exact: true }),
        ).toHaveCount(0);
        await expect(
          dialog.getByRole("button", { name: "Start Task", exact: true }),
          "a task needing a key can't start without one",
        ).toBeDisabled();
        await page.keyboard.press("Escape");
        await expect(dialog).toBeHidden();

        // The template: it names the key that was removed.
        await openTemplate(page, workflow, template.id);
        await expect(
          page.getByText(
            `${credential.title} was removed. Choose a connection to keep this workflow running.`,
          ),
          "the template should say its key was removed",
        ).toBeVisible();
        const startFromTemplate = page.waitForResponse(
          isTemplateStart(template.id),
        );
        await page
          .getByRole("button", { name: "Start task from template" })
          .click();
        const refused = await startFromTemplate;
        observe(
          testInfo,
          `starting the template anyway answers ${refused.status()}: ` +
            (await refused.text()).slice(0, 300),
        );
        expect(refused.status()).toBe(400);
        // The toast dismisses itself after 5 s, so only its showing is checked;
        // the refusal's text is recorded above, from the response.
        await expect(page.getByText("Failed to start task")).toBeVisible();
        expect(await listRunIDs(page, workflow)).toEqual([firstRunID]);

        // The builder: the step names the key that was removed.
        const scrape = await readScrapeStep(page, workflow);
        await page.goto(`/build?flowID=${workflow.graphID}`);
        await new BuildPage(page).closeTutorial();
        const step = page.locator(`[data-id="custom-node-${scrape.nodeID}"]`);
        await expect(step).toBeVisible({ timeout: 20_000 });
        await expect(
          step.getByText(`${credential.title} was removed.`, { exact: false }),
          "the builder should say the step's key was removed",
        ).toBeVisible();
      } finally {
        await deleteSchedule(page, schedule.id);
      }
    } finally {
      await deleteCredential(page, credential);
    }
  });
});

// ── Steps only this journey takes ──────────────────────────────────────

async function saveInBuilder(page: Page, workflow: Workflow) {
  const saved = page.waitForResponse(
    (response) =>
      response.request().method() === "PUT" &&
      new URL(response.url()).pathname.endsWith(
        `/api/graphs/${workflow.graphID}`,
      ),
  );
  await page.getByTestId("save-control-save-button").click();
  const name = page.getByTestId("save-control-name-input");
  await name.waitFor({ state: "visible" });
  await page.getByTestId("save-control-save-agent-button").click();
  const response = await saved;
  expect(response.ok(), `saving in the builder: ${response.status()}`).toBe(
    true,
  );
  return readScrapeStep(page, workflow);
}

async function saveRunAsTemplate(page: Page, runID: string, name: string) {
  const row = page.getByTestId(`library-run-item-${runID}`);
  await expect(row.getByRole("img", { name: "Task completed" })).toBeVisible();
  await row.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: "Save as template" }).click();
  const dialog = page.getByRole("dialog", { name: "Create Template" });
  await dialog.getByRole("textbox", { name: "Name", exact: true }).fill(name);
  const created = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.endsWith("/api/library/presets"),
  );
  await dialog.getByRole("button", { name: "Create Template" }).click();
  const response = await created;
  expect(response.ok(), `saving the template: ${response.status()}`).toBe(true);
  await expect(dialog).toBeHidden();
  return (await response.json()) as {
    id: string;
    credentials: Record<string, { id: string }>;
  };
}

async function openTemplate(
  page: Page,
  workflow: Workflow,
  templateID: string,
) {
  await page.goto(
    `/library/agents/${workflow.libraryAgentID}?activeTab=templates&activeItem=${templateID}`,
  );
  await waitForAgentPageLoad(page);
  await expect(
    page.getByRole("button", { name: "Start task from template" }),
  ).toBeVisible();
}

async function startedRunID(response: {
  ok(): boolean;
  status(): number;
  json(): Promise<unknown>;
}) {
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  return ((await response.json()) as { id: string }).id;
}

// A brand-new user, signed in, in a browser context of their own that gets
// what the journey's context gets: its user agent and flags, the changelog
// marked seen, and the guard, whose findings fail the journey.
async function asSomeoneElse(
  browser: Browser,
  {
    baseURL,
    flagOverrides,
  }: { baseURL: string | undefined; flagOverrides: Record<string, unknown> },
  body: (person: Person) => Promise<void>,
) {
  const context = await browser.newContext({
    baseURL,
    bypassCSP: true,
    userAgent: JOURNEY_USER_AGENT,
  });
  const guard = watchForUnexpectedErrors(context, []);
  try {
    await markChangelogSeen(context);
    await serveHostedFlags(context, flagOverrides);
    const page = await context.newPage();
    await body({ page, user: await provisionNewJourneyUser(page) });
    const problems = await guard.collect();
    expect(
      problems,
      `the other user hit ${problems.length} unexpected error(s):\n\n` +
        problems.join("\n\n"),
    ).toEqual([]);
  } finally {
    await context.close();
  }
}
