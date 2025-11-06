// Note: all the comments with //(number)! are for the docs
//ignore them when reading the code, but if you change something,
//make sure to update the docs! Your autoformmater will break this page,
// so don't run it on this file.
// --8<-- [start:BuildPageExample]

import test from "@playwright/test";
import { BuildPage } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl } from "./utils/assertion";
import { getTestUser } from "./utils/auth";

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
    await buildPage.closeTutorial();
  });

  // Helper function to add blocks starting with a specific letter, split into parts for parallelization
  async function addBlocksStartingWithSplit(letter: string, part: number, totalParts: number): Promise<void> {
    const blockIdsToSkip = await buildPage.getBlocksToSkip();
    const blockTypesToSkip = ["Input", "Output", "Agent", "AI"];
    const targetLetter = letter.toLowerCase();
    
    const allBlocks = await buildPage.getFilteredBlocksFromAPI(block => 
      block.name[0].toLowerCase() === targetLetter &&
      !blockIdsToSkip.includes(block.id) && 
      !blockTypesToSkip.includes(block.type)
    );

    const blocksToAdd = allBlocks.filter((_, index) => 
      index % totalParts === (part - 1)
    );

    console.log(`Adding ${blocksToAdd.length} blocks starting with "${letter}" (part ${part}/${totalParts})`);
    
    for (const block of blocksToAdd) {
      await buildPage.addBlock(block);
    }
  }

  // Reason Ignore: admonishment is in the wrong place visually with correct prettier rules
  // prettier-ignore
  test("user can add a block", async ({ page: _page }) => { //(6)!
    await buildPage.openBlocksPanel(); //(10)!
    const blocks = await buildPage.getFilteredBlocksFromAPI(block => block.name[0].toLowerCase() === "a");
    const block = blocks.at(-1);
    if (!block) throw new Error("No block found");

    await buildPage.addBlock(block); //(11)!
    await buildPage.closeBlocksPanel(); //(12)!
    await buildPage.hasBlock(block); //(13)!
  });
  // --8<-- [end:BuildPageExample]

  test("user can add blocks starting with a (part 1)", async () => {
    await addBlocksStartingWithSplit("a", 1, 2);
  });

  test("user can add blocks starting with a (part 2)", async () => {
    await addBlocksStartingWithSplit("a", 2, 2);
  });

  test("user can add blocks starting with b", async () => {
    await addBlocksStartingWithSplit("b", 1, 1);
  });

  test("user can add blocks starting with c", async () => {
    await addBlocksStartingWithSplit("c", 1, 1);
  });

  test("user can add blocks starting with d", async () => {
    await addBlocksStartingWithSplit("d", 1, 1);
  });

  test("user can add blocks starting with e", async () => {
    test.setTimeout(60000); // Increase timeout for many Exa blocks
    await addBlocksStartingWithSplit("e", 1, 2);
  });

  test("user can add blocks starting with e pt 2", async () => {
    test.setTimeout(60000); // Increase timeout for many Exa blocks
    await addBlocksStartingWithSplit("e", 2, 2);
  });

  test("user can add blocks starting with f", async () => {
    await addBlocksStartingWithSplit("f", 1, 1);
  });

  test("user can add blocks starting with g (part 1)", async () => {
    await addBlocksStartingWithSplit("g", 1, 3);
  });

  test("user can add blocks starting with g (part 2)", async () => {
    await addBlocksStartingWithSplit("g", 2, 3);
  });

  test("user can add blocks starting with g (part 3)", async () => {
    await addBlocksStartingWithSplit("g", 3, 3);
  });

  test("user can add blocks starting with h", async () => {
    await addBlocksStartingWithSplit("h", 1, 1);
  });

  test("user can add blocks starting with i", async () => {
    await addBlocksStartingWithSplit("i", 1, 1);
  });

  test("user can add blocks starting with j", async () => {
    await addBlocksStartingWithSplit("j", 1, 1);
  });

  test("user can add blocks starting with k", async () => {
    await addBlocksStartingWithSplit("k", 1, 1);
  });

  test("user can add blocks starting with l", async () => {
    await addBlocksStartingWithSplit("l", 1, 1);
  });

  test("user can add blocks starting with m", async () => {
    await addBlocksStartingWithSplit("m", 1, 1);
  });

  test("user can add blocks starting with n", async () => {
    await addBlocksStartingWithSplit("n", 1, 1);
  });

  test("user can add blocks starting with o", async () => {
    await addBlocksStartingWithSplit("o", 1, 1);
  });

  test("user can add blocks starting with p", async () => {
    await addBlocksStartingWithSplit("p", 1, 1);
  });

  test("user can add blocks starting with q", async () => {
    await addBlocksStartingWithSplit("q", 1, 1);
  });

  test("user can add blocks starting with r", async () => {
    await addBlocksStartingWithSplit("r", 1, 1);
  });

  test("user can add blocks starting with s (part 1)", async () => {
    await addBlocksStartingWithSplit("s", 1, 3);
  });

  test("user can add blocks starting with s (part 2)", async () => {
    await addBlocksStartingWithSplit("s", 2, 3);
  });

  test("user can add blocks starting with s (part 3)", async () => {
    await addBlocksStartingWithSplit("s", 3, 3);
  });

  test("user can add blocks starting with t", async () => {
    await addBlocksStartingWithSplit("t", 1, 1);
  });

  test("user can add blocks starting with u", async () => {
    await addBlocksStartingWithSplit("u", 1, 1);
  });

  test("user can add blocks starting with v", async () => {
    await addBlocksStartingWithSplit("v", 1, 1);
  });

  test("user can add blocks starting with w", async () => {
    await addBlocksStartingWithSplit("w", 1, 1);
  });

  test("user can add blocks starting with x", async () => {
    await addBlocksStartingWithSplit("x", 1, 1);
  });

  test("user can add blocks starting with y", async () => {
    await addBlocksStartingWithSplit("y", 1, 1);
  });

  test("user can add blocks starting with z", async () => {
    await addBlocksStartingWithSplit("z", 1, 1);
  });

  test("build navigation is accessible from navbar", async ({ page }) => {
    // Navigate somewhere else first
    await page.goto("/marketplace"); //(4)!

    // Check that navigation to the Builder is available on the page
    await buildPage.navbar.clickBuildLink();

    await hasUrl(page, "/build");
    await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();
  });

  test("user can add two blocks and connect them", async ({ page }) => {
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
  });

  test.skip("user can build an agent with inputs and output blocks", async ({ page }, testInfo) => {
    test.setTimeout(testInfo.timeout * 10);

    // prep
    await buildPage.openBlocksPanel();

    // Get input block from Input category
    const inputBlocks = await buildPage.getBlocksForCategory("Input");
    const inputBlock = inputBlocks.find((b) => b.name === "Agent Input");
    if (!inputBlock) throw new Error("Input block not found");
    await buildPage.addBlock(inputBlock);

    // Get output block from Output category  
    const outputBlocks = await buildPage.getBlocksForCategory("Output");
    const outputBlock = outputBlocks.find((b) => b.name === "Agent Output");
    if (!outputBlock) throw new Error("Output block not found");
    await buildPage.addBlock(outputBlock);

    // Get calculator block from Logic category
    const logicBlocks = await buildPage.getBlocksForCategory("Logic");
    const calculatorBlock = logicBlocks.find((b) => b.name === "Calculator");
    if (!calculatorBlock) throw new Error("Calculator block not found");
    await buildPage.addBlock(calculatorBlock);

    await buildPage.closeBlocksPanel();

    // Wait for blocks to be fully loaded
    await page.waitForTimeout(1000);

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

    // await buildPage.runAgent();
    // await buildPage.fillRunDialog({
    //   Value: "10",
    // });
    // await buildPage.clickRunDialogRunButton();
    // await buildPage.waitForCompletionBadge();
    // await test
    //   .expect(buildPage.isCompletionBadgeVisible())
    //   .resolves.toBeTruthy();
  });
});
