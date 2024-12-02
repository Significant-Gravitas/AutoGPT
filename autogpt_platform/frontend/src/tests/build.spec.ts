// profile.spec.ts
import { test } from "./fixtures";
import { BuildPage } from "./pages/build.page";

test.describe("Build", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page, loginPage, testUser }, testInfo) => {
    buildPage = new BuildPage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");
    await buildPage.navbar.clickBuildLink();
  });

  test("user can add a block", async ({ page }) => {
    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL(new RegExp("/.*build"));
    await buildPage.closeTutorial();
    await buildPage.openBlocksPanel();
    const block = {
      id: "31d1064e-7446-4693-a7d4-65e5ca1180d1",
      name: "Add to Dictionary",
      description: "Add to Dictionary",
    };
    await buildPage.addBlock(block);
    await buildPage.closeBlocksPanel();
    await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
  });

  test("user can add all blocks", async ({ page }, testInfo) => {
    // this test is slow af so we 10x the timeout (sorry future me)
    await test.setTimeout(testInfo.timeout * 10);
    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL(new RegExp("/.*build"));
    await buildPage.closeTutorial();
    await buildPage.openBlocksPanel();
    const blocks = await buildPage.getBlocks();

    // add all the blocks in order
    for (const block of blocks) {
      await buildPage.addBlock(block);
    }
    await buildPage.closeBlocksPanel();
    // check that all the blocks are visible
    for (const block of blocks) {
      await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
    }
    // fill in the input for the agent input block
    await buildPage.fillBlockInputByPlaceholder(
      blocks.find((b) => b.name === "Agent Input")?.id ?? "",
      "Enter Name",
      "Agent Input Field",
    );
    await buildPage.fillBlockInputByPlaceholder(
      blocks.find((b) => b.name === "Agent Output")?.id ?? "",
      "Enter Name",
      "Agent Output Field",
    );
    // check that we can save the agent with all the blocks
    await buildPage.saveAgent("all blocks test", "all blocks test");
    // page should have a url like http://localhost:3000/build?flowID=f4f3a1da-cfb3-430f-a074-a455b047e340
    await test.expect(page).toHaveURL(new RegExp("/.*build\\?flowID=.+"));
  });

  test("build navigation is accessible from navbar", async ({ page }) => {
    await buildPage.navbar.clickBuildLink();
    await test.expect(page).toHaveURL(new RegExp("/build"));
    // workaround for #8788
    await page.reload();
    await page.reload();
    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
  });

  test("user can add two blocks and connect them", async ({
    page,
  }, testInfo) => {
    await test.setTimeout(testInfo.timeout * 10);

    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL(new RegExp("/.*build"));
    await buildPage.closeTutorial();
    await buildPage.openBlocksPanel();

    // Define the blocks to add
    const block1 = {
      id: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
      name: "Store Value 1",
      description: "Store Value Block 1",
    };
    const block2 = {
      id: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
      name: "Store Value 2",
      description: "Store Value Block 2",
    };

    // Add the blocks
    await buildPage.addBlock(block1);
    await buildPage.addBlock(block2);
    await buildPage.closeBlocksPanel();

    // Connect the blocks
    await buildPage.connectBlockOutputToBlockInput(
      "1-1-output-source",
      "Output Source",
      "1-2-input-target",
      "Input Target",
    );

    // Fill in the input for the first block
    await buildPage.fillBlockInputByPlaceholder(
      block1.id,
      "Enter input",
      "Test Value",
      "1",
    );

    // Save the agent and wait for the URL to update
    await buildPage.saveAgent(
      "Connected Blocks Test",
      "Testing block connections",
    );
    await test.expect(page).toHaveURL(new RegExp("/.*build\\?flowID=.+"));

    // Wait for the save button to be enabled again
    await page.waitForSelector(
      '[data-testid="blocks-control-save-button"]:not([disabled])',
    );

    // Ensure the run button is enabled
    const runButton = page.locator('[data-id="primary-action-run-agent"]');
    await test.expect(runButton).toBeEnabled();

    // Run the agent
    await runButton.click();

    // Wait for processing to complete by checking the completion badge
    await page.waitForSelector('[data-id^="badge-"][data-id$="-COMPLETED"]');

    // Get the first completion badge and verify it's visible
    const completionBadge = page
      .locator('[data-id^="badge-"][data-id$="-COMPLETED"]')
      .first();
    await test.expect(completionBadge).toBeVisible();
  });
});
