import path from "path";
import type { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage, createUniqueAgentName } from "./pages/build.page";
import {
  clickRunButton,
  dismissFeedbackDialog,
  getActiveItemId,
  importAgentFromFile,
  LibraryPage,
} from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.library });
test.describe.configure({ mode: "serial" });

const TEST_AGENT_PATH = path.resolve(__dirname, "assets", "testing_agent.json");
const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";

function createLongRunningCalculatorGraph(
  agentName: string,
  calculatorCount: number = 150,
) {
  const nodes = Array.from({ length: calculatorCount }, (_, index) => ({
    id: `calc-${index + 1}`,
    block_id: CALCULATOR_BLOCK_ID,
    input_default:
      index === 0
        ? {
            operation: "Add",
            a: 1,
            b: 1,
            round_result: false,
          }
        : {
            operation: "Add",
            b: 1,
            round_result: false,
          },
    metadata: {
      position: { x: 320 * index, y: 120 },
    },
    input_links: [],
    output_links: [],
  }));

  const links = Array.from({ length: calculatorCount - 1 }, (_, index) => ({
    source_id: `calc-${index + 1}`,
    sink_id: `calc-${index + 2}`,
    source_name: "result",
    sink_name: "a",
  }));

  nodes.push({
    id: "final-output",
    block_id: AGENT_OUTPUT_BLOCK_ID,
    input_default: {
      name: "Final result",
      description: "Long-running calculator chain output",
    },
    metadata: {
      position: { x: 320 * calculatorCount, y: 120 },
    },
    input_links: [],
    output_links: [],
  });
  links.push({
    source_id: `calc-${calculatorCount}`,
    sink_id: "final-output",
    source_name: "result",
    sink_name: "value",
  });

  return {
    name: agentName,
    description:
      "Deterministic long-running calculator chain for runner stop coverage",
    is_active: true,
    nodes,
    links,
  };
}

async function createLongRunningSavedAgent(
  page: Page,
  agentName: string,
): Promise<void> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      graph: createLongRunningCalculatorGraph(agentName),
      source: "upload",
    },
  });
  expect(response.ok(), "expected graph creation API request to succeed").toBe(
    true,
  );

  const body = (await response.json()) as {
    id?: string;
    version?: number;
    data?: { id?: string; version?: number };
  };
  expect(
    body.data?.id ?? body.id,
    "graph creation should return a graph id",
  ).toBeTruthy();
}

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

  // Register the popup listener before clicking so we don't miss a fast open.
  // A short timeout covers the case where the link opens in the current tab.
  const popupPromise = page
    .context()
    .waitForEvent("page", { timeout: 10000 })
    .catch(() => null);
  await libraryPage.clickOpenInBuilder(importedAgent);
  const builderPage = (await popupPromise) ?? page;

  await builderPage.waitForLoadState("domcontentloaded");
  await expect(builderPage).toHaveURL(/\/build/);
  const importedBuildPage = new BuildPage(builderPage);
  await importedBuildPage.waitForNodeOnCanvas();
  expect(await importedBuildPage.getNodeCount()).toBeGreaterThan(0);
  if (builderPage !== page) {
    await builderPage.close();
  }
});

test("library happy path: user can start and stop a saved task from runner UI", async ({
  page,
}) => {
  test.setTimeout(180000);

  const agentName = createUniqueAgentName("E2E Stop Task Agent");
  await createLongRunningSavedAgent(page, agentName);

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);

  const setupTaskButton = page.getByRole("button", {
    name: /Setup your task/i,
  });
  await expect(setupTaskButton).toBeVisible({ timeout: 15000 });
  await setupTaskButton.click();

  const startTaskButton = page
    .getByRole("button", { name: /Start Task/i })
    .first();
  await expect(startTaskButton).toBeVisible({ timeout: 15000 });
  await expect(startTaskButton).toBeEnabled({ timeout: 15000 });
  await startTaskButton.click();

  await expect
    .poll(() => getActiveItemId(page), { timeout: 45000 })
    .not.toBe(null);
  await expect
    .poll(() => libraryPage.getRunStatus(), { timeout: 45000 })
    .toBe("running");

  const stopTaskButton = page.getByRole("button", { name: /Stop task/i });
  await expect(stopTaskButton).toBeVisible({ timeout: 30000 });
  await stopTaskButton.click();

  await expect
    .poll(() => libraryPage.getRunStatus(), { timeout: 45000 })
    .toBe("terminated");
});

test("library happy path: user can rerun a completed task from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } =
    await buildPage.createAndSaveSimpleAgent("E2E Rerun Agent");

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);
  await libraryPage.waitForRunToComplete();
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

  await libraryPage.waitForRunToComplete();

  // Simple agent has no AgentOutputBlock — verify run completion only.
  const runStatus = await libraryPage.getRunStatus();
  expect(runStatus).toBe("completed");
});

test("library happy path: user can delete a completed task from the run sidebar", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } = await buildPage.createAndSaveSimpleAgent(
    "E2E Delete Task Agent",
  );

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);
  await libraryPage.waitForRunToComplete();
  await dismissFeedbackDialog(page);

  // Open the per-task actions dropdown ("More actions" three-dot button)
  // and use the menu's Delete task option to remove the run.
  const moreActionsButton = page
    .getByRole("button", { name: "More actions" })
    .first();
  await expect(moreActionsButton).toBeVisible({ timeout: 15000 });
  await moreActionsButton.click();

  await page.getByRole("menuitem", { name: /Delete( this)? task/i }).click();

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

test("library happy path: user can open the agent in builder from the exact runner customise-agent path", async ({
  page,
  context,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { agentName } = await buildPage.createAndSaveSimpleAgent(
    "E2E View Task Agent",
  );

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);
  await libraryPage.waitForRunToComplete();
  await dismissFeedbackDialog(page);

  // The "View task details" eye-icon button on a completed run opens the
  // agent in the builder in a new tab. This exercises the runner → builder
  // navigation that QA item #22 ("Customise Agent" from Runner UI) covers.
  const selectedRunId = getActiveItemId(page);
  expect(selectedRunId).toBeTruthy();

  const viewTaskButton = page
    .locator('[aria-label="View task details"]')
    .first();
  await expect(viewTaskButton).toBeVisible({ timeout: 15000 });
  const customiseAgentHref = await viewTaskButton.getAttribute("href");
  expect(customiseAgentHref).toContain("flowID=");
  expect(customiseAgentHref).toContain("flowVersion=");
  expect(customiseAgentHref).toContain(`flowExecutionID=${selectedRunId}`);

  const popupPromise = context.waitForEvent("page", { timeout: 15000 });
  await viewTaskButton.click();
  const builderTab = await popupPromise;

  await builderTab.waitForLoadState("domcontentloaded");
  await expect(builderTab).toHaveURL(/\/build/);
  await expect(builderTab).toHaveURL(
    new RegExp(`flowExecutionID=${selectedRunId}`),
  );

  // Verify the builder canvas actually rendered with the agent's nodes —
  // a navigation that lands on /build but never paints the graph would
  // otherwise pass on URL alone.
  const builderTabPage = new BuildPage(builderTab);
  await builderTabPage.waitForNodeOnCanvas();
  expect(await builderTabPage.getNodeCount()).toBeGreaterThan(0);

  await builderTab.close();
});
