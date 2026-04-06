import path from "path";
import { Page } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { SMOKE_AUTH_STATES } from "../credentials/accounts";
import {
  clickRunButton,
  getRunStatus,
  navigateToAgentByName,
  waitForAgentPageLoad,
  waitForRunToComplete,
} from "../pages/library.page";
import { LibraryPage } from "../pages/library.page";

test.use({ storageState: SMOKE_AUTH_STATES.library });

const TEST_AGENT_PATH = path.resolve(
  __dirname,
  "..",
  "assets",
  "testing_agent.json",
);

function createUniqueAgentName(prefix: string) {
  return `${prefix} ${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function importAgentIntoLibrary(page: Page, prefix: string) {
  const libraryPage = new LibraryPage(page);
  const importedAgentName = createUniqueAgentName(prefix);

  await page.goto("/library");
  await libraryPage.openUploadDialog();
  await libraryPage.fillUploadForm(
    importedAgentName,
    "PR smoke library coverage",
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

test("@smoke library happy path: user can import an agent file into Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "Smoke Import Agent",
  );

  expect(importedAgent.name).toContain("Smoke Import Agent");
});

test("@smoke library happy path: user can open the imported or saved agent from Library in builder", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { libraryPage, importedAgent } = await importAgentIntoLibrary(
    page,
    "Smoke Open Agent",
  );

  const [builderPage] = await Promise.all([
    page.context().waitForEvent("page"),
    libraryPage.clickOpenInBuilder(importedAgent),
  ]);

  await builderPage.waitForLoadState("domcontentloaded");
  await expect(builderPage).toHaveURL(/\/build/);
  await builderPage.close();
});

test("@smoke library happy path: user can run a saved or imported agent from Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { importedAgent } = await importAgentIntoLibrary(
    page,
    "Smoke Run Import Agent",
  );

  await navigateToAgentByName(page, importedAgent.name);
  await waitForAgentPageLoad(page);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(["completed", "failed", "running"]).toContain(runStatus);
});
