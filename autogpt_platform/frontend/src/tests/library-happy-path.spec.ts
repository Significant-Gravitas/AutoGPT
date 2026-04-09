import path from "path";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage, createUniqueAgentName } from "./pages/build.page";
import {
  assertRunProducedOutput,
  clickRunButton,
  dismissFeedbackDialog,
  getActiveItemId,
  getRunStatus,
  importAgentFromFile,
  openSavedAgentInLibrary,
  waitForRunToComplete,
} from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.library });
test.describe.configure({ mode: "serial" });

const TEST_AGENT_PATH = path.resolve(__dirname, "assets", "testing_agent.json");

test("library happy path: user can import an agent file into Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentFromFile(
    page,
    TEST_AGENT_PATH,
    createUniqueAgentName("E2E Import Agent"),
  );

  expect(importedAgent.name).toContain("E2E Import Agent");
});

test("library happy path: user can open the imported or saved agent from Library in builder", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { libraryPage, importedAgent } = await importAgentFromFile(
    page,
    TEST_AGENT_PATH,
    createUniqueAgentName("E2E Open Agent"),
  );

  const popupPromise = page
    .context()
    .waitForEvent("page")
    .catch(() => null);
  await libraryPage.clickOpenInBuilder(importedAgent);
  const builderPage = (await popupPromise) ?? page;

  await builderPage.waitForLoadState("domcontentloaded");
  await expect(builderPage).toHaveURL(/\/build/);
  if (builderPage !== page) {
    await builderPage.close();
  }
});

test("library happy path: user can rerun a completed task from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } =
    await buildPage.createAndSaveSimpleAgent("E2E Rerun Agent");

  await openSavedAgentInLibrary(page, agentName);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);
  await dismissFeedbackDialog(page);

  const rerunTaskButton = page.getByRole("button", { name: /Rerun task/i });
  await expect(rerunTaskButton).toBeVisible({ timeout: 45000 });

  await expect
    .poll(() => getActiveItemId(page), { timeout: 45000 })
    .not.toBe(null);

  const initialRunId = getActiveItemId(page);
  expect(initialRunId).toBeTruthy();

  await rerunTaskButton.click();

  await expect(page.getByText("Run started", { exact: true })).toBeVisible({
    timeout: 15000,
  });

  await expect
    .poll(() => getActiveItemId(page), { timeout: 45000 })
    .not.toBe(initialRunId);

  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(runStatus).toBe("completed");
  await assertRunProducedOutput(page);
});

test("library happy path: user can delete a completed task from the run sidebar", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } = await buildPage.createAndSaveSimpleAgent(
    "E2E Delete Task Agent",
  );

  await openSavedAgentInLibrary(page, agentName);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);
  await dismissFeedbackDialog(page);

  // Open the per-task actions dropdown ("More actions" three-dot button)
  // and use the menu's Delete task option to remove the run.
  const moreActionsButton = page
    .getByRole("button", { name: "More actions" })
    .first();
  await expect(moreActionsButton).toBeVisible({ timeout: 15000 });
  await moreActionsButton.click();

  await page.getByRole("menuitem", { name: "Delete task" }).click();

  const confirmDialog = page.getByRole("dialog", { name: /Delete task/i });
  await expect(confirmDialog).toBeVisible({ timeout: 10000 });
  await confirmDialog.getByRole("button", { name: /^Delete Task$/ }).click();

  // Toast confirms the backend actually deleted (not just dialog closed).
  await expect(page.getByText("Task deleted", { exact: true })).toBeVisible({
    timeout: 15000,
  });

  // Sidebar should drop the only run, returning the page to the empty
  // "Setup your task" state.
  await expect(
    page.getByRole("button", { name: /Setup your task/i }),
  ).toBeVisible({ timeout: 15000 });
});

test("library happy path: user can open the agent in builder from the run detail (View task details)", async ({
  page,
  context,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } = await buildPage.createAndSaveSimpleAgent(
    "E2E View Task Agent",
  );

  await openSavedAgentInLibrary(page, agentName);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);
  await dismissFeedbackDialog(page);

  // The "View task details" eye-icon button on a completed run opens the
  // agent in the builder in a new tab. This exercises the runner → builder
  // navigation that QA item #22 ("Customise Agent" from Runner UI) covers.
  const viewTaskButton = page.getByRole("button", {
    name: "View task details",
  });
  await expect(viewTaskButton).toBeVisible({ timeout: 15000 });

  const popupPromise = context.waitForEvent("page", { timeout: 15000 });
  await viewTaskButton.click();
  const builderTab = await popupPromise;

  await builderTab.waitForLoadState("domcontentloaded");
  await expect(builderTab).toHaveURL(/\/build/);

  // Verify the builder canvas actually rendered with the agent's nodes —
  // a navigation that lands on /build but never paints the graph would
  // otherwise pass on URL alone.
  const builderTabPage = new BuildPage(builderTab);
  await builderTabPage.waitForNodeOnCanvas();
  expect(await builderTabPage.getNodeCount()).toBeGreaterThan(0);

  await builderTab.close();
});
