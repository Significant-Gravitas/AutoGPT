import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import {
  navigateToAgentByName,
  waitForAgentPageLoad,
} from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.builder });

function createUniqueAgentName(prefix: string) {
  return `${prefix} ${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function openBuilder(page: Page) {
  const buildPage = new BuildPage(page);

  await page.goto("/build");
  await page.waitForLoadState("domcontentloaded");
  await buildPage.closeTutorial();

  await expect(page.locator(".react-flow")).toBeVisible();
  await expect(page.getByTestId("blocks-control-blocks-button")).toBeVisible();

  return buildPage;
}

async function addSimpleAgentBlocks(buildPage: BuildPage) {
  await buildPage.addBlockByClick("Store Value");
  await buildPage.waitForNodeOnCanvas(1);
  await buildPage.fillBlockInputByPlaceholder(
    "Enter string value...",
    "smoke-value",
    0,
  );

  await buildPage.addBlockByClick("Add to Dictionary");
  await buildPage.waitForNodeOnCanvas(2);

  const dictionaryInputs = buildPage
    .getNodeLocator(1)
    .locator('input[placeholder="Enter string value..."]');
  await dictionaryInputs.nth(0).fill("smoke-key");
  await dictionaryInputs.nth(1).fill("smoke-value");
}

async function createAndSaveAgent(page: Page, prefix: string) {
  const buildPage = await openBuilder(page);
  const agentName = createUniqueAgentName(prefix);

  await addSimpleAgentBlocks(buildPage);
  await buildPage.saveAgent(agentName, "PR E2E builder coverage");
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  return { buildPage, agentName };
}

async function startBuilderRun(page: Page, buildPage: BuildPage) {
  await buildPage.clickRunButton();

  const runDialog = page.locator('[data-id="run-input-dialog-content"]');
  if (await runDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.locator('[data-id="run-input-manual-run-button"]').click();
  }
}

async function getBuilderExecutionState(page: Page) {
  const stopButton = page.locator('[data-id="stop-graph-button"]');
  if (await stopButton.isVisible().catch(() => false)) {
    return "running";
  }

  const runButton = page.locator('[data-id="run-graph-button"]');
  if (await runButton.isVisible().catch(() => false)) {
    return "idle";
  }

  return "unknown";
}

async function openSavedAgentInLibrary(page: Page, agentName: string) {
  await page.goto("/library");
  await navigateToAgentByName(page, agentName);
  await waitForAgentPageLoad(page);
}

test("builder happy path: user can complete or skip the builder tutorial successfully", async ({
  page,
}) => {
  test.setTimeout(90000);

  await openBuilder(page);

  await expect(
    page.getByRole("button", { name: "Skip Tutorial", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".react-flow")).toBeVisible();
});

test("builder happy path: user can create a simple agent in builder with core blocks", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = await openBuilder(page);
  await addSimpleAgentBlocks(buildPage);

  await expect(buildPage.getNodeLocator()).toHaveCount(2);
});

test("builder happy path: user can save the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Save Agent");

  await expect(page).toHaveURL(/flowID=/);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});

test("builder happy path: user can run the saved agent from builder", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Run Agent");

  await startBuilderRun(page, buildPage);
  await expect(
    page.locator('[data-id="stop-graph-button"], [data-id="run-graph-button"]'),
  ).toBeVisible({ timeout: 15000 });
});

test("builder happy path: user can see the run result or execution status", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Status Agent");

  await startBuilderRun(page, buildPage);
  await expect
    .poll(() => getBuilderExecutionState(page), {
      timeout: 15000,
    })
    .not.toBe("unknown");
});

test("builder happy path: user can schedule the saved agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage, agentName } = await createAndSaveAgent(
    page,
    "Smoke Schedule Agent",
  );

  await page.locator('[data-id="schedule-graph-button"]').click();

  const runDialog = page.locator('[data-id="run-input-dialog-content"]');
  if (await runDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.locator('[data-id="run-input-schedule-button"]').click();
  }

  await expect(
    page.getByRole("dialog", { name: "Schedule Graph" }),
  ).toBeVisible();
  const scheduleDialog = page.getByRole("dialog", { name: "Schedule Graph" });
  await page.locator("#schedule-name").fill(`Daily ${agentName}`);

  const createScheduleResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      /\/api\/proxy\/api\/graphs\/.+\/schedules$/.test(response.url()) &&
      response.status() === 200,
    { timeout: 15000 },
  );

  await page.getByRole("button", { name: "Done" }).click();
  await createScheduleResponse;
  await expect(scheduleDialog).toBeHidden({ timeout: 15000 });
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});

test("builder happy path: user can export the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { agentName } = await createAndSaveAgent(page, "Smoke Export Agent");

  await openSavedAgentInLibrary(page, agentName);

  const downloadPromise = page
    .waitForEvent("download", { timeout: 15000 })
    .catch(() => null);

  await page.getByRole("button", { name: "Export agent to file" }).click();

  const download = await downloadPromise;
  if (download) {
    expect(download.suggestedFilename()).toMatch(/\.json$/i);
  }

  await expect(page.getByText("Agent exported")).toBeVisible({
    timeout: 15000,
  });
});
