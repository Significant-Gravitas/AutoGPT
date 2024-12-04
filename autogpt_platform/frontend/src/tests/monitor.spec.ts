import { test } from "./fixtures";
import { BuildPage } from "./pages/build.page";
import { MonitorPage } from "./pages/monitor.page";
import { v4 as uuidv4 } from "uuid";

test.describe("Monitor", () => {
  let buildPage: BuildPage;
  let monitorPage: MonitorPage;

  test.beforeEach(async ({ page, loginPage, testUser }, testInfo) => {
    buildPage = new BuildPage(page);
    monitorPage = new MonitorPage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");

    // add a test agent
    const basicBlock = await buildPage.getBasicBlock();
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
  });

  test("user can view agents", async ({ page }) => {
    const agents = await monitorPage.listAgents();
    // there should be at least one agent
    await test.expect(agents.length).toBeGreaterThan(0);
  });

  // test("user can export agents", async ({ page }) => {
  //   // await monitorPage.selectAgent({
  //   //   id: "test-agent-1",
  //   //   name: "test-agent-1",
  //   // });
  //   await monitorPage.exportAgent();
  // });
  test("user can view runs", async ({ page }) => {
    const runs = await monitorPage.listRuns();
    console.log(runs);
    // there should be at least one run
    await test.expect(runs.length).toBeGreaterThan(0);
  });
});
