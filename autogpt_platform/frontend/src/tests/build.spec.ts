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
});
