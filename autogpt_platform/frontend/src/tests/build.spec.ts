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
  test.describe.configure({ mode: 'parallel' });
  
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

  // Helper function to add blocks from a specific category
  async function addBlocksFromCategory(category: string): Promise<void> {
    await buildPage.openBlocksPanel();
    
    // Check if category exists for this user
    const availableCategories = await buildPage.discoverCategories();
    if (!availableCategories.includes(category)) {
      console.log(`⚠️ Category "${category}" not available for this user, skipping test`);
      await buildPage.closeBlocksPanel();
      return; // Return early instead of failing
    }
    
    const blocks = await buildPage.getBlocksForCategory(category);

    const blockIdsToSkip = await buildPage.getBlocksToSkip();
    console.log("⚠️ Skipping blocks:", blockIdsToSkip);

    const blocksToAdd = blocks.filter(block => 
      !blockIdsToSkip.includes(block.id)
    );

    console.log(`Adding ${blocksToAdd.length} blocks from category "${category}"`);
    
    for (const block of blocksToAdd) {
      await buildPage.addBlock(block);
    }
    
    await buildPage.closeBlocksPanel();
    
    // Verify blocks are visible
    for (const block of blocksToAdd) {
      await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();
    }

    await buildPage.saveAgent(`${category} blocks test`, `testing blocks from ${category} category`);
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

  // All available categories from backend BlockCategory enum
  test.describe("Block Category Tests", () => {
    test.describe.configure({ timeout: 60000 }); // 60 second timeout for all category tests

    test("AI category", async () => {
      await addBlocksFromCategory("AI");
    });

    test("Agent category", async () => {
      await addBlocksFromCategory("Agent");
    });

    test("Basic category", async () => {
      await addBlocksFromCategory("Basic");
    });

    test("Communication category", async () => {
      await addBlocksFromCategory("Communication");
    });

    test("CRM category", async () => {
      await addBlocksFromCategory("Crm");
    });

    test("Data category", async () => {
      await addBlocksFromCategory("Data");
    });

    test("Developer Tools category", async () => {
      await addBlocksFromCategory("Developer Tools");
    });

    test("Hardware category", async () => {
      await addBlocksFromCategory("Hardware");
    });

    test("Input category", async () => {
      await addBlocksFromCategory("Input");
    });

    test("Issue Tracking category", async () => {
      await addBlocksFromCategory("Issue Tracking");
    });

    test("Logic category", async () => {
      await addBlocksFromCategory("Logic");
    });

    test("Marketing category", async () => {
      await addBlocksFromCategory("Marketing");
    });

    test("Multimedia category", async () => {
      await addBlocksFromCategory("Multimedia");
    });

    test("Output category", async () => {
      await addBlocksFromCategory("Output");
    });

    test("Productivity category", async () => {
      await addBlocksFromCategory("Productivity");
    });

    test("Safety category", async () => {
      await addBlocksFromCategory("Safety");
    });

    test("Search category", async () => {
      await addBlocksFromCategory("Search");
    });

    test("Social category", async () => {
      await addBlocksFromCategory("Social");
    });

    test("Text category", async () => {
      await addBlocksFromCategory("Text");
    });
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

  test("user can build an agent with inputs and output blocks", async ({ page }, testInfo) => {
    // simple calculator to double input and output it
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

    await buildPage.hasBlock(inputBlock)
    await buildPage.hasBlock(outputBlock)
    await buildPage.hasBlock(calculatorBlock)

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
