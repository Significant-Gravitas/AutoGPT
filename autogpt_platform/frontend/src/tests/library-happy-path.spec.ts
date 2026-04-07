import { randomUUID } from "crypto";
import path from "path";
import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
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

async function fillVisibleTaskInputs(page: Page) {
  const seededEmail = getSeededTestUser("smokeMarketplace").email;
  const inputs = page.locator(
    'input:visible:not([type="hidden"]):not([type="file"]):not([disabled]), textarea:visible:not([disabled])',
  );
  const inputCount = await inputs.count();

  for (let index = 0; index < inputCount; index += 1) {
    const input = inputs.nth(index);
    const currentValue = await input.inputValue().catch(() => "");
    if (currentValue.trim()) {
      continue;
    }

    const type = (await input.getAttribute("type"))?.toLowerCase() ?? "text";
    const placeholder = (
      (await input.getAttribute("placeholder")) ?? ""
    ).toLowerCase();
    const ariaLabel = (
      (await input.getAttribute("aria-label")) ?? ""
    ).toLowerCase();
    const labelText = `${placeholder} ${ariaLabel}`;

    if (type === "checkbox" || type === "radio") {
      continue;
    }

    const value =
      type === "email" || labelText.includes("email")
        ? seededEmail
        : type === "number"
          ? "1"
          : "e2e-input";

    await input.fill(value).catch(() => {});
  }
}

async function openNewTaskDialog(page: Page) {
  const setupTaskButton = page.getByRole("button", {
    name: /Setup your task/i,
  });
  if (await setupTaskButton.isVisible({ timeout: 5000 }).catch(() => false)) {
    await setupTaskButton.click();
    return;
  }

  const newTaskButton = page.getByRole("button", { name: /^New task$/i });
  await expect(newTaskButton).toBeVisible({ timeout: 15000 });
  await newTaskButton.click();
}

async function createScheduleFromLibraryAgentPage(page: Page): Promise<string> {
  await openNewTaskDialog(page);

  const scheduleTaskButton = page.getByRole("button", {
    name: /Schedule Task/i,
  });
  await expect(scheduleTaskButton).toBeVisible({ timeout: 15000 });
  if (!(await scheduleTaskButton.isEnabled().catch(() => false))) {
    await fillVisibleTaskInputs(page);
  }
  await expect(scheduleTaskButton).toBeEnabled({ timeout: 15000 });
  await scheduleTaskButton.click();

  const scheduleNameInput = page.locator("#schedule-name");
  await expect(scheduleNameInput).toBeVisible({ timeout: 15000 });
  const scheduleName = (await scheduleNameInput.inputValue()).trim();
  expect(scheduleName.length).toBeGreaterThan(0);

  const scheduleButton = page.getByRole("button", { name: /^Schedule$/i });
  await expect(scheduleButton).toBeEnabled({ timeout: 15000 });
  const successToast = page.getByText("Schedule created", { exact: true });
  const failureToast = page.getByText(/Failed to create schedule/i);

  async function getScheduleSubmissionState() {
    if (await successToast.isVisible().catch(() => false)) {
      return "success";
    }

    if (await failureToast.isVisible().catch(() => false)) {
      return "failed";
    }

    if (!(await scheduleButton.isVisible().catch(() => false))) {
      return "success";
    }

    return "pending";
  }

  for (let attempt = 0; attempt < 2; attempt += 1) {
    await scheduleButton.click({ force: true });

    const submitted = await expect
      .poll(getScheduleSubmissionState, { timeout: 10000 })
      .not.toBe("pending")
      .then(() => true)
      .catch(() => false);

    if (submitted) {
      break;
    }
  }

  await expect
    .poll(getScheduleSubmissionState, { timeout: 20000 })
    .toBe("success");

  await expect
    .poll(() => new URL(page.url()).searchParams.get("activeTab"), {
      timeout: 15000,
    })
    .toBe("scheduled");

  await expect(
    page.getByText(scheduleName, { exact: true }).first(),
  ).toBeVisible({
    timeout: 15000,
  });

  return scheduleName;
}

async function reloadLibraryAgentPage(page: Page) {
  await page.reload();
  await waitForAgentPageLoad(page);
  await page.waitForLoadState("networkidle").catch(() => undefined);
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

test("library happy path: user can create and delete a schedule from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = await createSavedAgent(page, "E2E Schedule Agent");

  await openLibraryAgentByName(page, agentName);
  const scheduleName = await createScheduleFromLibraryAgentPage(page);

  await reloadLibraryAgentPage(page);
  await expect(
    page.getByText(scheduleName, { exact: true }).first(),
  ).toBeVisible({
    timeout: 15000,
  });

  const deleteScheduleButton = page.getByRole("button", {
    name: /Delete schedule/i,
  });
  await expect(deleteScheduleButton).toBeVisible({ timeout: 15000 });
  await deleteScheduleButton.click();

  const deleteScheduleDialog = page.getByRole("dialog", {
    name: /Delete schedule/i,
  });
  await expect(deleteScheduleDialog).toBeVisible({ timeout: 15000 });
  await deleteScheduleDialog
    .getByRole("button", { name: /Delete Schedule/i })
    .click();

  await expect(page.getByText("Schedule deleted", { exact: true })).toBeVisible(
    {
      timeout: 15000,
    },
  );
  await expect(
    page.getByRole("button", { name: /Setup your task/i }),
  ).toBeVisible({
    timeout: 15000,
  });
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
