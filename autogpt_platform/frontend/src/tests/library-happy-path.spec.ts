import { randomUUID } from "crypto";
import path from "path";
import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import {
  clickRunButton,
  getRunStatus,
  navigateToAgentByName,
  waitForAgentPageLoad,
  waitForRunToComplete,
} from "./pages/library.page";
import { LibraryPage } from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.library });
test.describe.configure({ mode: "serial" });

const TEST_AGENT_PATH = path.resolve(__dirname, "assets", "testing_agent.json");

function createUniqueAgentName(prefix: string) {
  return `${prefix} ${Date.now()}-${randomUUID().slice(0, 8)}`;
}

const ACCEPTED_RUN_STATUSES = [
  "completed",
  "failed",
  "running",
  "queued",
  "review",
] as const;

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

async function createSavedAgent(page: Page, prefix: string) {
  const buildPage = new BuildPage(page);
  const agentName = createUniqueAgentName(prefix);

  await page.goto("/build");
  await page.waitForLoadState("domcontentloaded");
  await buildPage.closeTutorial();
  await expect(page.locator(".react-flow")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("blocks-control-blocks-button")).toBeVisible({
    timeout: 15000,
  });

  await addSimpleAgentBlocks(buildPage);
  await buildPage.saveAgent(agentName, "PR E2E library coverage");
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  return agentName;
}

async function importAgentIntoLibrary(page: Page, prefix: string) {
  const libraryPage = new LibraryPage(page);
  const importedAgentName = createUniqueAgentName(prefix);

  await page.goto("/library");
  await libraryPage.openUploadDialog();
  await libraryPage.fillUploadForm(
    importedAgentName,
    "PR E2E library coverage",
  );

  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(TEST_AGENT_PATH);
  await expect(page.getByRole("button", { name: "Upload" })).toBeEnabled({
    timeout: 10000,
  });
  await page.getByRole("button", { name: "Upload" }).click();

  await expect(page).toHaveURL(/\/build/);

  await page.goto("/library");
  await libraryPage.searchAgents(importedAgentName);
  await libraryPage.waitForAgentsToLoad();

  const importedAgents = await libraryPage.getAgents();
  const importedAgent = importedAgents.find((agent) =>
    agent.name.includes(importedAgentName),
  );

  expect(importedAgent).toBeTruthy();
  if (!importedAgent) {
    throw new Error("Imported agent was not found in the library");
  }

  return { libraryPage, importedAgent };
}

function getActiveItemId(page: Page) {
  return new URL(page.url()).searchParams.get("activeItem");
}

async function openLibraryAgentByName(page: Page, agentName: string) {
  const libraryPage = new LibraryPage(page);

  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);
  await libraryPage.waitForAgentsToLoad();
  await navigateToAgentByName(page, agentName);
  await waitForAgentPageLoad(page);
}

async function dismissFeedbackDialog(page: Page) {
  const feedbackDialog = page.getByRole("dialog", {
    name: "We'd love your feedback",
  });
  if (!(await feedbackDialog.isVisible().catch(() => false))) {
    return;
  }

  const cancelButton = feedbackDialog.getByRole("button", { name: "Cancel" });
  if (await cancelButton.isVisible().catch(() => false)) {
    await cancelButton.click();
    await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
    return;
  }

  await feedbackDialog.getByRole("button", { name: "Close" }).click();
  await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
}

test("library happy path: user can import an agent file into Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "E2E Import Agent",
  );

  expect(importedAgent.name).toContain("E2E Import Agent");
});

test("library happy path: user can open the imported or saved agent from Library in builder", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { libraryPage, importedAgent } = await importAgentIntoLibrary(
    page,
    "E2E Open Agent",
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

test("library happy path: user can run a saved or imported agent from Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = await createSavedAgent(page, "E2E Run Agent");

  await openLibraryAgentByName(page, agentName);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(ACCEPTED_RUN_STATUSES).toContain(
    runStatus as (typeof ACCEPTED_RUN_STATUSES)[number],
  );
});

test("library happy path: user can rerun a completed task from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = await createSavedAgent(page, "E2E Rerun Agent");

  await openLibraryAgentByName(page, agentName);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);
  await dismissFeedbackDialog(page);

  const rerunTaskButton = page.getByRole("button", { name: /Rerun task/i });
  await expect(rerunTaskButton).toBeVisible({ timeout: 45000 });

  await expect
    .poll(() => getActiveItemId(page), {
      timeout: 45000,
    })
    .not.toBe(null);

  const initialRunId = getActiveItemId(page);
  expect(initialRunId).toBeTruthy();

  await rerunTaskButton.click();

  await expect(page.getByText("Run started", { exact: true })).toBeVisible({
    timeout: 15000,
  });

  await expect
    .poll(() => getActiveItemId(page), {
      timeout: 45000,
    })
    .not.toBe(initialRunId);

  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(ACCEPTED_RUN_STATUSES).toContain(
    runStatus as (typeof ACCEPTED_RUN_STATUSES)[number],
  );
});
