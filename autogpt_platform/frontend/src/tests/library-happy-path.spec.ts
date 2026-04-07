import { randomUUID } from "crypto";
import path from "path";
import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import {
  clickRunButton,
  getRunStatus,
  navigateToAgentByName,
  waitForAgentPageLoad,
  waitForRunToComplete,
} from "./pages/library.page";
import { LibraryPage } from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.library });

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

async function createScheduleFromLibraryAgentPage(
  page: Page,
  scheduleName: string,
) {
  await openNewTaskDialog(page);

  const scheduleTaskButton = page.getByRole("button", {
    name: /Schedule Task/i,
  });
  await expect(scheduleTaskButton).toBeVisible({ timeout: 15000 });
  await scheduleTaskButton.click();

  const scheduleNameInput = page.locator("#schedule-name");
  await expect(scheduleNameInput).toBeVisible({ timeout: 15000 });
  await scheduleNameInput.fill(scheduleName);

  const scheduleButton = page.getByRole("button", { name: /^Schedule$/i });
  await expect(scheduleButton).toBeEnabled({ timeout: 15000 });
  await scheduleButton.click();

  await expect(page.getByText("Schedule created", { exact: true })).toBeVisible(
    {
      timeout: 15000,
    },
  );

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
}

async function reloadLibraryAgentPage(page: Page) {
  await page.reload();
  await waitForAgentPageLoad(page);
  await page.waitForLoadState("networkidle").catch(() => undefined);
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

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "E2E Run Import Agent",
  );

  await navigateToAgentByName(page, importedAgent.name);
  await waitForAgentPageLoad(page);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(ACCEPTED_RUN_STATUSES).toContain(
    runStatus as (typeof ACCEPTED_RUN_STATUSES)[number],
  );
});

test("library happy path: user can create, edit, and delete a schedule from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "E2E Schedule Agent",
  );

  const scheduleName = createUniqueAgentName("E2E Library Schedule");
  const updatedScheduleName = createUniqueAgentName(
    "E2E Updated Library Schedule",
  );

  await navigateToAgentByName(page, importedAgent.name);
  await waitForAgentPageLoad(page);
  await createScheduleFromLibraryAgentPage(page, scheduleName);

  await reloadLibraryAgentPage(page);
  await expect(
    page.getByText(scheduleName, { exact: true }).first(),
  ).toBeVisible({
    timeout: 15000,
  });

  const editScheduleButton = page.getByRole("button", {
    name: /Edit schedule/i,
  });
  await expect(editScheduleButton).toBeVisible({ timeout: 15000 });
  await editScheduleButton.click();

  const scheduleNameInput = page.locator("#schedule-name");
  await expect(scheduleNameInput).toBeVisible({ timeout: 15000 });
  await scheduleNameInput.fill(updatedScheduleName);

  const scheduleTimeInput = page.locator("#schedule-time");
  await expect(scheduleTimeInput).toBeVisible({ timeout: 15000 });
  await scheduleTimeInput.fill("10:15");

  await page.getByRole("button", { name: /^Save$/i }).click();
  await expect(scheduleNameInput).toBeHidden({ timeout: 15000 });
  await expect(
    page.getByText(updatedScheduleName, { exact: true }).first(),
  ).toBeVisible({
    timeout: 15000,
  });

  await reloadLibraryAgentPage(page);
  await expect(
    page.getByText(updatedScheduleName, { exact: true }).first(),
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
  await expect(page.getByText(/Nothing scheduled yet/i)).toBeVisible({
    timeout: 15000,
  });
});

test("library happy path: user can rerun a completed task from the Library agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "E2E Rerun Agent",
  );

  await navigateToAgentByName(page, importedAgent.name);
  await waitForAgentPageLoad(page);
  await clickRunButton(page);

  await expect
    .poll(() => getActiveItemId(page), {
      timeout: 45000,
    })
    .not.toBe(null);

  const initialRunId = getActiveItemId(page);
  expect(initialRunId).toBeTruthy();

  const rerunTaskButton = page.getByRole("button", { name: /Rerun task/i });
  await expect(rerunTaskButton).toBeVisible({ timeout: 45000 });
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
