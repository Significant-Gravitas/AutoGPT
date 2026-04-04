import { test, expect } from "./coverage-fixture";
import { BuildPage } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl } from "./utils/assertion";
import { getTestUser } from "./utils/auth";

test.describe("Builder", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page }) => {
    test.setTimeout(60000);
    const loginPage = new LoginPage(page);
    const testUser = await getTestUser();

    buildPage = new BuildPage(page);

    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await hasUrl(page, "/marketplace");

    await page.goto("/build");
    await page.waitForLoadState("domcontentloaded");
    await buildPage.closeTutorial();
  });

  // --- Core tests ---

  test("build page loads successfully", async () => {
    await expect(buildPage.isLoaded()).resolves.toBeTruthy();
    await expect(
      buildPage.getPlaywrightPage().getByTestId("blocks-control-blocks-button"),
    ).toBeVisible();
    await expect(
      buildPage.getPlaywrightPage().getByTestId("save-control-save-button"),
    ).toBeVisible();
  });

  test("user can add a block via block menu", async () => {
    const initialCount = await buildPage.getNodeCount();
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(initialCount + 1);
    expect(await buildPage.getNodeCount()).toBe(initialCount + 1);
  });

  test("user can add multiple blocks", async () => {
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(2);

    expect(await buildPage.getNodeCount()).toBe(2);
  });

  test("user can remove a block", async () => {
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    // Deselect, then re-select the node and delete
    await buildPage.clickCanvas();
    await buildPage.selectNode(0);
    await buildPage.deleteSelectedNodes();

    await expect(buildPage.getNodeLocator()).toHaveCount(0, { timeout: 5000 });
  });

  test("user can save an agent", async ({ page }) => {
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    await buildPage.saveAgent("E2E Test Agent", "Created by e2e test");
    await buildPage.waitForSaveComplete();

    expect(page.url()).toContain("flowID=");
  });

  test("user can save and run button becomes enabled", async () => {
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    await buildPage.saveAgent("Runnable Agent", "Test run button");
    await buildPage.waitForSaveComplete();
    await buildPage.waitForSaveButton();

    await expect(buildPage.isRunButtonEnabled()).resolves.toBeTruthy();
  });

  // --- Copy / Paste test ---

  test("user can copy and paste a node", async ({ context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);

    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    await buildPage.selectNode(0);
    await buildPage.copyViaKeyboard();
    await buildPage.pasteViaKeyboard();

    await buildPage.waitForNodeOnCanvas(2);
    expect(await buildPage.getNodeCount()).toBe(2);
  });

  // --- Run agent test ---

  test("user can run an agent from the builder", async () => {
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);

    // Save the agent (required before running)
    await buildPage.saveAgent("Run Test Agent", "Testing run from builder");
    await buildPage.waitForSaveComplete();
    await buildPage.waitForSaveButton();

    // Click run button
    await buildPage.clickRunButton();

    // Either the run dialog appears or the agent starts running directly
    const runDialogOrRunning = await Promise.race([
      buildPage
        .getPlaywrightPage()
        .locator('[data-id="run-input-dialog-content"]')
        .waitFor({ state: "visible", timeout: 10000 })
        .then(() => "dialog"),
      buildPage
        .getPlaywrightPage()
        .locator('[data-id="stop-graph-button"]')
        .waitFor({ state: "visible", timeout: 10000 })
        .then(() => "running"),
    ]).catch(() => "timeout");

    expect(["dialog", "running"]).toContain(runDialogOrRunning);
  });
});
