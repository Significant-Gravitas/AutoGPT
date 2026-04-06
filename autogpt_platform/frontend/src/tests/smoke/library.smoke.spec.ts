import path from "path";
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

test("@smoke library flow: user can import an agent, open it in builder, and run it from the library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const libraryPage = new LibraryPage(page);
  const importedAgentName = `Smoke Import Agent ${Date.now()}`;
  const testAgentPath = path.resolve(
    __dirname,
    "..",
    "assets",
    "testing_agent.json",
  );

  await page.goto("/library");
  await libraryPage.openUploadDialog();
  await libraryPage.fillUploadForm(
    importedAgentName,
    "PR smoke library coverage",
  );

  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(testAgentPath);
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

  const [builderPage] = await Promise.all([
    page.context().waitForEvent("page"),
    libraryPage.clickOpenInBuilder(importedAgent),
  ]);
  await builderPage.waitForLoadState("domcontentloaded");
  await expect(builderPage).toHaveURL(/\/build/);
  await builderPage.close();

  await navigateToAgentByName(page, importedAgent.name);
  await waitForAgentPageLoad(page);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(["completed", "failed"]).toContain(runStatus);
});
