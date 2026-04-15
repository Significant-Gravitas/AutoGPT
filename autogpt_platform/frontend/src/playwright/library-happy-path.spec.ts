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

const TEST_AGENT_PATH = path.resolve(__dirname, "assets", "testing_agent.json");
const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const STOPPED_RUN_STATUSES = new Set([
  "terminated",
  "failed",
  "incomplete",
  "completed",
]);

type UploadedGraphNode = {
  id: string;
  block_id: string;
  input_default: Record<string, unknown>;
  metadata: {
    position: {
      x: number;
      y: number;
    };
  };
  input_links: unknown[];
  output_links: unknown[];
};

function createLongRunningCalculatorGraph(
  agentName: string,
  calculatorCount: number = 150,
) {
  const nodes: UploadedGraphNode[] = Array.from(
    { length: calculatorCount },
    (_, index) => ({
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
    }),
  );

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
): Promise<{ graphId: string; graphVersion: number }> {
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

  return {
    graphId: String(body.data?.id ?? body.id),
    graphVersion: Number(body.data?.version ?? body.version ?? 1),
  };
}

async function createDeterministicCalculatorSavedAgent(
  page: Page,
  agentName: string,
  outputName: string,
): Promise<void> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      graph: {
        name: agentName,
        description:
          "Deterministic calculator output for run-result assertions",
        is_active: true,
        nodes: [
          {
            id: "calc-1",
            block_id: CALCULATOR_BLOCK_ID,
            input_default: {
              operation: "Add",
              a: 1,
              b: 1,
              round_result: false,
            },
            metadata: {
              position: { x: 120, y: 160 },
            },
            input_links: [],
            output_links: [],
          },
          {
            id: "final-output",
            block_id: AGENT_OUTPUT_BLOCK_ID,
            input_default: {
              name: outputName,
              description: "Deterministic result output",
            },
            metadata: {
              position: { x: 520, y: 160 },
            },
            input_links: [],
            output_links: [],
          },
        ],
        links: [
          {
            source_id: "calc-1",
            sink_id: "final-output",
            source_name: "result",
            sink_name: "value",
          },
        ],
      },
      source: "upload",
    },
  });
  expect(
    response.ok(),
    "expected deterministic calculator graph creation API request to succeed",
  ).toBe(true);
}

async function getExecutionStatusFromApi(
  page: Page,
  graphId: string,
  runId: string,
): Promise<string> {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphId}/executions/${runId}`,
  );
  expect(response.ok(), "execution details API should succeed").toBe(true);

  const body = (await response.json()) as { status?: string };
  return body.status?.toLowerCase() ?? "unknown";
}

async function createAndSaveDeterministicOutputAgent(
  page: Page,
  prefix: string,
): Promise<{ agentName: string; expectedOutput: string; outputName: string }> {
  const buildPage = new BuildPage(page);
  const agentName = createUniqueAgentName(prefix);
  const expectedOutput = `e2e-output-${Date.now()}`;
  const outputName = `e2e-result-${Date.now()}`;

  await buildPage.open();
  await buildPage.addBlockByClick("Store Value");
  await buildPage.waitForNodeOnCanvas(1);
  await buildPage.fillBlockInputByPlaceholder(
    "Enter string value...",
    expectedOutput,
    0,
  );

  await buildPage.addBlockByClick("Agent Output");
  await buildPage.waitForNodeOnCanvas(2);
  await buildPage.connectNodes(0, 1);
  await buildPage.fillLastNodeTextInput("Agent Output", outputName);

  await buildPage.saveAgent(
    agentName,
    "Deterministic output agent for library run verification",
  );
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  return { agentName, expectedOutput, outputName };
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
  const { graphId } = await createLongRunningSavedAgent(page, agentName);

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);

  await expect
    .poll(() => getActiveItemId(page), { timeout: 45000 })
    .not.toBe(null);
  const runId = getActiveItemId(page);
  expect(runId, "run id should be present after starting task").toBeTruthy();
  await expect
    .poll(() => libraryPage.getRunStatus(), { timeout: 45000 })
    .toBe("running");

  const stopTaskButton = page.getByRole("button", { name: /Stop task/i });
  await expect(stopTaskButton).toBeVisible({ timeout: 30000 });
  const stopResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response
        .url()
        .includes(`/api/graphs/${graphId}/executions/${runId}/stop`),
    { timeout: 15000 },
  );
  await stopTaskButton.click();
  const stopResponse = await stopResponsePromise;

  expect(stopResponse.ok(), "stop run API should succeed").toBe(true);
  await expect(page.getByText("Run stopped")).toBeVisible({ timeout: 15000 });
  await expect
    .poll(
      async () => {
        const status = await getExecutionStatusFromApi(
          page,
          graphId,
          String(runId),
        );
        return STOPPED_RUN_STATUSES.has(status) ? status : "running";
      },
      { timeout: 45000 },
    )
    .not.toBe("running");
});

test("library happy path: user can run a saved agent and verify expected output", async ({
  page,
}) => {
  test.setTimeout(150000);

  const agentName = createUniqueAgentName("E2E Expected Output Agent");
  const outputName = `e2e-result-${Date.now()}`;
  await createDeterministicCalculatorSavedAgent(page, agentName, outputName);

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);
  await libraryPage.waitForRunToComplete();
  await dismissFeedbackDialog(page);

  await libraryPage.assertRunProducedOutput();
  await libraryPage.assertRunOutputValue(outputName, /^2(?:\.0+)?$/);
  await expect
    .poll(() => libraryPage.getRunStatus(), { timeout: 15000 })
    .toBe("completed");
});

test("library happy path: user can edit a saved agent from Library and keep changes after refresh", async ({
  page,
}) => {
  test.setTimeout(150000);

  const { agentName } = await createAndSaveDeterministicOutputAgent(
    page,
    "E2E Edit Persist Agent",
  );
  const editedValue = `edited-value-${Date.now()}`;

  const libraryPage = new LibraryPage(page);
  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);
  await libraryPage.waitForAgentsToLoad();

  const agentCard = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(agentCard).toBeVisible({ timeout: 15000 });

  const popupPromise = page
    .context()
    .waitForEvent("page", { timeout: 10000 })
    .catch(() => null);
  await agentCard
    .getByTestId("library-agent-card-open-in-builder-link")
    .first()
    .click();
  const builderPage = (await popupPromise) ?? page;

  const builderTabPage = new BuildPage(builderPage);
  await builderTabPage.waitForNodeOnCanvas();
  await builderTabPage.fillBlockInputByPlaceholder(
    "Enter string value...",
    editedValue,
    0,
  );

  await builderPage.getByTestId("save-control-save-button").click();
  const saveAgentButton = builderPage.getByRole("button", {
    name: "Save Agent",
  });
  if (await saveAgentButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await expect(saveAgentButton).toBeEnabled({ timeout: 10000 });
    await saveAgentButton.click();
    await expect(saveAgentButton).toBeHidden({ timeout: 15000 });
  }

  await builderPage.reload();
  await builderTabPage.waitForNodeOnCanvas();
  await expect(
    builderTabPage
      .getNodeLocator(0)
      .locator('input[placeholder="Enter string value..."]'),
  ).toHaveValue(editedValue);

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

  // Sidebar should drop the only run, returning the page to initial
  // task-entry state.
  await expect(
    page.getByRole("button", { name: /^(Setup your task|New task)$/i }),
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
