import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";

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
