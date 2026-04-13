import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import { LibraryPage } from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.builder });

test("builder happy path: user can walk through the builder tutorial and cancel midway, persisting canceled state", async ({
  page,
}) => {
  test.setTimeout(180000);

  const buildPage = new BuildPage(page);
  await buildPage.startTutorial();
  await buildPage.walkWelcomeToBlockMenu();
  await buildPage.walkSearchAndAddCalculator();
  await buildPage.cancelTutorial();

  expect(await buildPage.getTutorialStateFromStorage()).toBe("canceled");
  expect(await buildPage.getNodeCount()).toBeGreaterThanOrEqual(1);
});

test("builder happy path: user can skip the builder tutorial from the welcome step", async ({
  page,
}) => {
  test.setTimeout(60000);

  const buildPage = new BuildPage(page);
  await buildPage.startTutorial();
  await buildPage.skipTutorialFromWelcome();
});

test("builder happy path: user can create a simple agent in builder with core blocks", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  await buildPage.open();
  await buildPage.addSimpleAgentBlocks();

  await expect(buildPage.getNodeLocator()).toHaveCount(2);
  await expect(
    buildPage
      .getNodeLocator(0)
      .locator('input[placeholder="Enter string value..."]'),
  ).toHaveValue("smoke-value");
  await expect(buildPage.getNodeTextInput("Add to Dictionary", 0)).toHaveValue(
    "smoke-key",
  );
  await expect(buildPage.getNodeTextInput("Add to Dictionary", 1)).toHaveValue(
    "smoke-value",
  );
});

test("builder happy path: user can save the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  await buildPage.createAndSaveSimpleAgent("Smoke Save Agent");

  await expect(page).toHaveURL(/flowID=/);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});

test("builder happy path: user can run the saved agent from builder and see execution state", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  await buildPage.createAndSaveSimpleAgent("Smoke Run Agent");

  await buildPage.startRun();
  await expect(
    page.locator('[data-id="stop-graph-button"], [data-id="run-graph-button"]'),
  ).toBeVisible({ timeout: 15000 });

  await expect
    .poll(() => buildPage.getExecutionState(), { timeout: 15000 })
    .not.toBe("unknown");
});

test("builder happy path: user can schedule the saved agent and run it from Library", async ({
  page,
}) => {
  test.setTimeout(180000);

  const buildPage = new BuildPage(page);
  const { agentName } = await buildPage.createAndSaveSimpleAgent(
    "Smoke Schedule Agent",
  );

  await buildPage.createScheduleForSavedAgent(agentName);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);

  const scheduledTab = page.getByRole("tab", { name: /Scheduled/i });
  await expect(scheduledTab).toBeVisible({ timeout: 15000 });
  await scheduledTab.click();

  const runNowButton = page.getByRole("button", { name: /Run now/i }).first();
  await expect(runNowButton).toBeVisible({ timeout: 15000 });
  await runNowButton.click();

  await libraryPage.waitForRunToComplete();

  // The simple agent (Store Value + Add to Dictionary) has no AgentOutputBlock,
  // so "No output from this run." is expected — assert only the run status.
  const runStatus = await libraryPage.getRunStatus();
  expect(runStatus).toBe("completed");
});

test("builder happy path: user can export the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName, graphId, graphVersion } =
    await buildPage.createAndSaveSimpleAgent("Smoke Export Agent");

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);

  const exportResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      response
        .url()
        .includes(
          `/api/proxy/api/graphs/${graphId}/versions/${graphVersion}`,
        ) &&
      response.url().includes("for_export=true"),
    { timeout: 15000 },
  );

  await libraryPage.clickExportAgent();
  const exportResponse = await exportResponsePromise;

  expect(exportResponse.ok(), "export request should succeed").toBe(true);
  const parsed = await exportResponse.json();

  expect(parsed.name, "exported agent must include name").toBeTruthy();
  expect(
    Array.isArray(parsed.nodes),
    "exported agent must have a nodes array",
  ).toBe(true);
  expect(
    parsed.nodes.length,
    "exported agent must contain at least one node",
  ).toBeGreaterThan(0);
  expect(
    Array.isArray(parsed.links),
    "exported agent must have a links array",
  ).toBe(true);

  await expect(page.getByText("Agent exported")).toBeVisible({
    timeout: 15000,
  });
});
