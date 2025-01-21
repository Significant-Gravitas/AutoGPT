import { expect, TestInfo } from "@playwright/test";
import { test } from "./fixtures";
import { BuildPage } from "./pages/build.page";
import { MonitorPage } from "./pages/monitor.page";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs/promises";
import path from "path";
// --8<-- [start:AttachAgentId]
test.describe("Monitor", () => {
  let buildPage: BuildPage;
  let monitorPage: MonitorPage;

  test.beforeEach(async ({ page, loginPage, testUser }, testInfo: TestInfo) => {
    buildPage = new BuildPage(page);
    monitorPage = new MonitorPage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");

    // add a test agent
    const basicBlock = await buildPage.getDictionaryBlockDetails();
    const id = uuidv4();
    await buildPage.createSingleBlockAgent(
      `test-agent-${id}`,
      `test-agent-description-${id}`,
      basicBlock,
    );
    await buildPage.runAgent();
    await monitorPage.navbar.clickMonitorLink();
    await monitorPage.waitForPageLoad();
    await test.expect(monitorPage.isLoaded()).resolves.toBeTruthy();
    testInfo.attach("agent-id", { body: id });
  });
  // --8<-- [end:AttachAgentId]

  test.afterAll(async ({}) => {
    // clear out the downloads folder
    console.log(
      `clearing out the downloads folder ${monitorPage.downloadsFolder}`,
    );

    await fs.rm(`${monitorPage.downloadsFolder}/monitor`, {
      recursive: true,
      force: true,
    });
  });

  test("user can view agents", async ({ page }) => {
    const agents = await monitorPage.listAgents();
    // there should be at least one agent
    await test.expect(agents.length).toBeGreaterThan(0);
  });

  test.skip("user can export and import agents", async ({
    page,
  }, testInfo: TestInfo) => {
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
    if (!agent) {
      throw new Error(`Agent ${testAttachName} not found`);
    }
    await monitorPage.exportToFile(agent);
    const download = await downloadPromise;

    // Wait for the download process to complete and save the downloaded file somewhere.
    await download.saveAs(
      `${monitorPage.downloadsFolder}/monitor/${download.suggestedFilename()}`,
    );
    console.log(`downloaded file to ${download.suggestedFilename()}`);
    await test.expect(download.suggestedFilename()).toBeDefined();
    // test-agent-uuid-v1.json
    await test.expect(download.suggestedFilename()).toContain("test-agent-");
    await test.expect(download.suggestedFilename()).toContain("v1.json");

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
    await monitorPage.waitForPageLoad();

    const postImportAgents = await monitorPage.listAgents();
    await test
      .expect(postImportAgents.length)
      .toBeGreaterThan(preImportAgents.length);
    console.log(`postImportAgents: ${JSON.stringify(postImportAgents)}`);
    const importedAgent = postImportAgents.find(
      (a: any) => a.name === `${baseName}-imported`,
    );
    await test.expect(importedAgent).toBeDefined();
  });

  test("user can view runs", async ({ page }) => {
    const runs = await monitorPage.listRuns();
    console.log(runs);
    // there should be at least one run
    await test.expect(runs.length).toBeGreaterThan(0);
  });
});
