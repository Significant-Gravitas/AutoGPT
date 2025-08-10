import test, { expect, TestInfo } from "@playwright/test";

import { BuildPage } from "./pages/build.page";
import { MonitorPage } from "./pages/monitor.page";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs/promises";
import path from "path";
import { LoginPage } from "./pages/login.page";
import { getTestUser } from "./utils/auth";
import { hasUrl } from "./utils/assertion";
import {
  navigateToLibrary,
  clickFirstAgent,
  runAgent,
  waitForAgentPageLoad,
} from "./pages/library.page";

test.describe.configure({
  mode: "parallel",
  timeout: 30000,
});
// --8<-- [start:AttachAgentId]
test.beforeEach(async ({ page }, testInfo: TestInfo) => {
  const loginPage = new LoginPage(page);
  const testUser = await getTestUser();
  const monitorPage = new MonitorPage(page);

  // Start each test with login using worker auth
  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  // Navigate to library and run the first agent
  await navigateToLibrary(page);
  await clickFirstAgent(page);
  await waitForAgentPageLoad(page);
  await runAgent(page);

  // Navigate to monitoring page
  await page.goto("/monitoring");
  await test.expect(monitorPage.isLoaded()).resolves.toBeTruthy();

  // Generate a test ID for tracking
  const id = uuidv4();
  testInfo.attach("agent-id", { body: id });
});
// --8<-- [end:AttachAgentId]

test.afterAll(async () => {
  // clear out the downloads folder
  const downloadsFolder = process.cwd() + "/downloads";
  console.log(`clearing out the downloads folder ${downloadsFolder}/monitor`);

  await fs.rm(`${downloadsFolder}/monitor`, {
    recursive: true,
    force: true,
  });
});

test.skip("user can export and import agents", async ({
  page,
}, testInfo: TestInfo) => {
  const monitorPage = new MonitorPage(page);
  const buildPage = new BuildPage(page);

  // --8<-- [start:ReadAgentId]
  if (testInfo.attachments.length === 0 || !testInfo.attachments[0].body) {
    throw new Error("No agent id attached to the test");
  }

  const testAttachName = testInfo.attachments[0].body.toString();
  // --8<-- [end:ReadAgentId]
  const agents = await monitorPage.listAgents();

  const downloadPromise = page.waitForEvent("download");

  const agent = agents.find(
    (a: any) => a.name === `test-agent-${testAttachName}`,
  );

  if (!agent) throw new Error(`Agent ${testAttachName} not found`);

  await monitorPage.exportToFile(agent);
  const download = await downloadPromise;

  // Wait for the download process to complete and save the downloaded file somewhere.
  await download.saveAs(
    `${monitorPage.downloadsFolder}/monitor/${download.suggestedFilename()}`,
  );

  console.log(`downloaded file to ${download.suggestedFilename()}`);

  expect(download.suggestedFilename()).toBeDefined();
  expect(download.suggestedFilename()).toContain("test-agent-");
  expect(download.suggestedFilename()).toContain("v1.json");

  // import the agent
  const preImportAgents = await monitorPage.listAgents();

  const filesInFolder = await fs.readdir(
    `${monitorPage.downloadsFolder}/monitor`,
  );

  const importFile = filesInFolder.find((f) => f.includes(testAttachName));
  if (!importFile) {
    throw new Error(`No import file found for agent ${testAttachName}`);
  }

  const baseName = importFile.split(".")[0];

  await monitorPage.importFromFile(
    path.resolve(monitorPage.downloadsFolder, "monitor"),
    importFile,
    baseName + "-imported",
  );

  // You'll be dropped at the build page, so hit run and then go back to monitor
  await buildPage.runAgent();
  await monitorPage.navbar.clickMonitorLink();

  const postImportAgents = await monitorPage.listAgents();

  expect(postImportAgents.length).toBeGreaterThan(preImportAgents.length);

  console.log(`postImportAgents: ${JSON.stringify(postImportAgents)}`);

  const importedAgent = postImportAgents.find(
    (a: any) => a.name === `${baseName}-imported`,
  );

  expect(importedAgent).toBeDefined();
});

test.skip("user can view runs and agents", async ({ page }) => {
  const monitorPage = new MonitorPage(page);
  // const runs = await monitorPage.listRuns();
  const agents = await monitorPage.listAgents();

  expect(agents.length).toBeGreaterThan(0);
});
