// Note: all the comments with //(number)! are for the docs
//ignore them when reading the code, but if you change something,
//make sure to update the docs! Your autoformmater will break this page,
// so don't run it on this file.
// --8<-- [start:BuildPageExample]

import test from "@playwright/test";
import { BuildPage } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { getTestUser } from "./utils/auth";
import { hasUrl } from "./utils/assertion";

// Reason Ignore: admonishment is in the wrong place visually with correct prettier rules
// prettier-ignore
test.describe("Build", () => { //(1)!
  let buildPage: BuildPage; //(2)!

  // Reason Ignore: admonishment is in the wrong place visually with correct prettier rules
  // prettier-ignore
  test.beforeEach(async ({ page }) => { //(3)! ts-ignore
    const loginPage = new LoginPage(page);
    const testUser = await getTestUser();

    buildPage = new BuildPage(page);

    // Start each test with login using worker auth
    await page.goto("/login"); //(4)!
    await loginPage.login(testUser.email, testUser.password);
    await hasUrl(page, "/marketplace"); //(5)!
    await buildPage.navbar.clickBuildLink();
    await hasUrl(page, "/build");
    await buildPage.waitForPageLoad();
    await buildPage.closeTutorial();
  });

  // Helper function to add blocks starting with a specific letter
  async function addBlocksStartingWith(letter: string, page: any): Promise<void> {
    await buildPage.openBlocksPanel();
    const blocks = await buildPage.getBlocks();

    const blockIdsToSkip = await buildPage.getBlocksToSkip();
    const blockTypesToSkip = ["Input", "Output", "Agent", "AI"];
    console.log("⚠️ Skipping blocks:", blockIdsToSkip);
    console.log("⚠️ Skipping block types:", blockTypesToSkip);

    const targetLetter = letter.toLowerCase();
    const blocksToAdd = blocks.filter(block => 
      block.name[0].toLowerCase() === targetLetter &&
      !blockIdsToSkip.includes(block.id) && 
      !blockTypesToSkip.includes(block.type)
    );

    console.log(`Adding ${blocksToAdd.length} blocks starting with "${letter}"`);
    
    for (const block of blocksToAdd) {
      await buildPage.addBlock(block);
    }
    
    await buildPage.closeBlocksPanel();
    
    // Verify blocks are visible
    for (const block of blocksToAdd) {
      await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
    }

    await buildPage.saveAgent(`blocks ${letter} test`, `testing blocks starting with ${letter}`);
    await test.expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
  }

  // Reason Ignore: admonishment is in the wrong place visually with correct prettier rules
  // prettier-ignore
  test("user can add a block", async ({ page: _page }) => { //(6)!
    await buildPage.openBlocksPanel(); //(10)!
    const block = await buildPage.getDictionaryBlockDetails();

    await buildPage.addBlock(block); //(11)!
    await buildPage.closeBlocksPanel(); //(12)!
    await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy(); //(13)!
  });
  // --8<-- [end:BuildPageExample]

  test("user can add blocks starting with a", async ({ page }) => {
    await addBlocksStartingWith("a", page);
  });

  test("user can add blocks starting with b", async ({ page }) => {
    await addBlocksStartingWith("b", page);
  });

  test("user can add blocks starting with c", async ({ page }) => {
    await addBlocksStartingWith("c", page);
  });

  test("user can add blocks starting with d", async ({ page }) => {
    await addBlocksStartingWith("d", page);
  });

  test("user can add blocks starting with e", async ({ page }) => {
    await addBlocksStartingWith("e", page);
  });

  test("user can add blocks starting with f", async ({ page }) => {
    await addBlocksStartingWith("f", page);
  });

  test("user can add blocks starting with g", async ({ page }) => {
    await addBlocksStartingWith("g", page);
  });

  test("user can add blocks starting with h", async ({ page }) => {
    await addBlocksStartingWith("h", page);
  });

  test("user can add blocks starting with i", async ({ page }) => {
    await addBlocksStartingWith("i", page);
  });

  test("user can add blocks starting with j", async ({ page }) => {
    await addBlocksStartingWith("j", page);
  });

  test("user can add blocks starting with k", async ({ page }) => {
    await addBlocksStartingWith("k", page);
  });

  test("user can add blocks starting with l", async ({ page }) => {
    await addBlocksStartingWith("l", page);
  });

  test("user can add blocks starting with m", async ({ page }) => {
    await addBlocksStartingWith("m", page);
  });

  test("user can add blocks starting with n", async ({ page }) => {
    await addBlocksStartingWith("n", page);
  });

  test("user can add blocks starting with o", async ({ page }) => {
    await addBlocksStartingWith("o", page);
  });

  test("user can add blocks starting with p", async ({ page }) => {
    await addBlocksStartingWith("p", page);
  });

  test("user can add blocks starting with q", async ({ page }) => {
    await addBlocksStartingWith("q", page);
  });

  test("user can add blocks starting with r", async ({ page }) => {
    await addBlocksStartingWith("r", page);
  });

  test("user can add blocks starting with s", async ({ page }) => {
    await addBlocksStartingWith("s", page);
  });

  test("user can add blocks starting with t", async ({ page }) => {
    await addBlocksStartingWith("t", page);
  });

  test("user can add blocks starting with u", async ({ page }) => {
    await addBlocksStartingWith("u", page);
  });

  test("user can add blocks starting with v", async ({ page }) => {
    await addBlocksStartingWith("v", page);
  });

  test("user can add blocks starting with w", async ({ page }) => {
    await addBlocksStartingWith("w", page);
  });

  test("user can add blocks starting with x", async ({ page }) => {
    await addBlocksStartingWith("x", page);
  });

  test("user can add blocks starting with y", async ({ page }) => {
    await addBlocksStartingWith("y", page);
  });

  test("user can add blocks starting with z", async ({ page }) => {
    await addBlocksStartingWith("z", page);
  });

  test.skip("user can add all blocks a-l", async ({ page }, testInfo) => {
    // this test is slow af so we 100x the timeout (sorry future me)
    test.setTimeout(testInfo.timeout * 100);

    await buildPage.openBlocksPanel();
    const blocks = await buildPage.getBlocks();

    const blockIdsToSkip = await buildPage.getBlocksToSkip();
    const blockTypesToSkip = ["Input", "Output", "Agent", "AI"];
    console.log("⚠️ Skipping blocks:", blockIdsToSkip);
    console.log("⚠️ Skipping block types:", blockTypesToSkip);

    // add all the blocks in order except for the agent executor block
    for (const block of blocks) {
      if (block.name[0].toLowerCase() >= "m") {
        continue;
      }
      if (!blockIdsToSkip.includes(block.id) && !blockTypesToSkip.includes(block.type)) {
        await buildPage.addBlock(block);
      }
    }
    await buildPage.closeBlocksPanel();
    // check that all the blocks are visible
    for (const block of blocks) {
      if (block.name[0].toLowerCase() >= "m") {
        continue;
      }
      if (!blockIdsToSkip.includes(block.id) && !blockTypesToSkip.includes(block.type)) {
        await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
      }
    }

    // check that we can save the agent with all the blocks
    await buildPage.saveAgent("all blocks test", "all blocks test");
    // page should have a url like http://localhost:3000/build?flowID=f4f3a1da-cfb3-430f-a074-a455b047e340
    await test.expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
  });

  test.skip("user can add all blocks m-z", async ({ page }, testInfo) => {
    // this test is slow af so we 100x the timeout (sorry future me)
    test.setTimeout(testInfo.timeout * 100);

    await buildPage.openBlocksPanel();
    const blocks = await buildPage.getBlocks();

    const blockIdsToSkip = await buildPage.getBlocksToSkip();
    const blockTypesToSkip = ["Input", "Output", "Agent", "AI"];
    console.log("⚠️ Skipping blocks:", blockIdsToSkip);
    console.log("⚠️ Skipping block types:", blockTypesToSkip);

    // add all the blocks in order except for the agent executor block
    for (const block of blocks) {
      if (block.name[0].toLowerCase() < "m") {
        continue;
      }
      if (!blockIdsToSkip.includes(block.id) && !blockTypesToSkip.includes(block.type)) {
        await buildPage.addBlock(block);
      }
    }
    await buildPage.closeBlocksPanel();
    // check that all the blocks are visible
    for (const block of blocks) {
      if (block.name[0].toLowerCase() < "m") {
        continue;
      }
      if (!blockIdsToSkip.includes(block.id) && !blockTypesToSkip.includes(block.type)) {
        await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
      }
    }

    // check that we can save the agent with all the blocks
    await buildPage.saveAgent("all blocks test", "all blocks test");
    // page should have a url like http://localhost:3000/build?flowID=f4f3a1da-cfb3-430f-a074-a455b047e340
    await test.expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
  });

  test("build navigation is accessible from navbar", async ({ page }) => {
    // Navigate somewhere else first
    await page.goto("/marketplace"); //(4)!

    // Check that navigation to the Builder is available on the page
    await buildPage.navbar.clickBuildLink();
    await buildPage.waitForPageLoad();

    await hasUrl(page, "/build");
    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
  });

  test("user can add two blocks and connect them", async ({ page }, testInfo) => {
    test.setTimeout(testInfo.timeout * 10);

    await buildPage.openBlocksPanel();

    // Define the blocks to add
    const block1 = {
      id: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
      name: "Store Value 1",
      description: "Store Value Block 1",
      type: "Standard",
    };
    const block2 = {
      id: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
      name: "Store Value 2",
      description: "Store Value Block 2",
      type: "Standard",
    };

    // Add the blocks
    await buildPage.addBlock(block1);
    await buildPage.addBlock(block2);
    await buildPage.closeBlocksPanel();

    // Connect the blocks
    await buildPage.connectBlockOutputToBlockInputViaDataId(
      "1-1-output-source",
      "1-2-input-target",
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
    await test.expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));

    // Wait for the save button to be enabled again
    await buildPage.waitForSaveButton();

    // Ensure the run button is enabled
    await test.expect(buildPage.isRunButtonEnabled()).resolves.toBeTruthy();

    // Run the agent
    await buildPage.runAgent();

    // Wait for processing to complete by checking the completion badge
    await buildPage.waitForCompletionBadge();

    // Get the first completion badge and verify it's visible
    await test
      .expect(buildPage.isCompletionBadgeVisible())
      .resolves.toBeTruthy();
  });

  test("user can build an agent with inputs and output blocks", async ({ page }) => {
    // simple calculator to double input and output it

    // prep
    await buildPage.openBlocksPanel();

    // find the blocks we want
    const blocks = await buildPage.getBlocks();
    const inputBlock = blocks.find((b) => b.name === "Agent Input");
    const outputBlock = blocks.find((b) => b.name === "Agent Output");
    const calculatorBlock = blocks.find((b) => b.name === "Calculator");
    if (!inputBlock || !outputBlock || !calculatorBlock) {
      throw new Error("Input or output block not found");
    }

    // add the blocks
    await buildPage.addBlock(inputBlock);
    await buildPage.addBlock(outputBlock);
    await buildPage.addBlock(calculatorBlock);
    await buildPage.closeBlocksPanel();

    // Wait for blocks to be fully loaded
    await page.waitForTimeout(1000);

    await test.expect(buildPage.hasBlock(inputBlock)).resolves.toBeTruthy();
    await test.expect(buildPage.hasBlock(outputBlock)).resolves.toBeTruthy();
    await test
      .expect(buildPage.hasBlock(calculatorBlock))
      .resolves.toBeTruthy();

    // Wait for blocks to be ready for connections
    await page.waitForTimeout(1000);

    await buildPage.connectBlockOutputToBlockInputViaName(
      inputBlock.id,
      "Result",
      calculatorBlock.id,
      "A",
    );
    await buildPage.connectBlockOutputToBlockInputViaName(
      inputBlock.id,
      "Result",
      calculatorBlock.id,
      "B",
    );
    await buildPage.connectBlockOutputToBlockInputViaName(
      calculatorBlock.id,
      "Result",
      outputBlock.id,
      "Value",
    );

    // Wait for connections to stabilize
    await page.waitForTimeout(1000);

    await buildPage.fillBlockInputByPlaceholder(
      inputBlock.id,
      "Enter Name",
      "Value",
    );
    await buildPage.fillBlockInputByPlaceholder(
      outputBlock.id,
      "Enter Name",
      "Doubled",
    );

    // Wait before changing dropdown
    await page.waitForTimeout(500);

    await buildPage.selectBlockInputValue(
      calculatorBlock.id,
      "Operation",
      "Add",
    );

    // Wait before saving
    await page.waitForTimeout(1000);

    await buildPage.saveAgent(
      "Input and Output Blocks Test",
      "Testing input and output blocks",
    );
    await test.expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));

    // Wait for save to complete
    await page.waitForTimeout(1000);

    await buildPage.runAgent();
    await buildPage.fillRunDialog({
      Value: "10",
    });
    await buildPage.clickRunDialogRunButton();
    await buildPage.waitForCompletionBadge();
    await test
      .expect(buildPage.isCompletionBadgeVisible())
      .resolves.toBeTruthy();
  });
});
