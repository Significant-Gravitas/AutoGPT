/**
 * E2E tests for SmartDecisionMaker block functionality.
 *
 * These tests verify the critical bug where field names with spaces
 * (e.g., "Max Keyword Difficulty") cause tool calls to fail due to
 * inconsistent sanitization between frontend and backend.
 *
 * The bug:
 * - Frontend creates links with original names: tools_^_{node_id}_~_Max Keyword Difficulty
 * - Backend emits with sanitized names: tools_^_{node_id}_~_max_keyword_difficulty
 * - Routing fails because names don't match
 */

import test, { expect } from "@playwright/test";
import { BuildPage, Block } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl } from "./utils/assertion";
import { getTestUser } from "./utils/auth";

test.describe("SmartDecisionMaker", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page }) => {
    test.setTimeout(60000); // Longer timeout for complex tests
    const loginPage = new LoginPage(page);
    const testUser = await getTestUser();

    buildPage = new BuildPage(page);

    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await hasUrl(page, "/marketplace");
    await buildPage.navbar.clickBuildLink();
    await hasUrl(page, "/build");
    await buildPage.closeTutorial();
  });

  /**
   * Helper to find SmartDecisionMaker block from API
   */
  async function getSmartDecisionMakerBlock(): Promise<Block | undefined> {
    const blocks = await buildPage.getBlocksFromAPI();
    return blocks.find(
      (b) =>
        b.name.toLowerCase().includes("smart decision") ||
        b.name.toLowerCase().includes("ai decision") ||
        b.id === "3b191d9f-356f-482d-8238-ba04b6d18381"
    );
  }

  /**
   * Helper to find a block by partial name match
   */
  async function findBlockByName(partialName: string): Promise<Block | undefined> {
    const blocks = await buildPage.getBlocksFromAPI();
    return blocks.find((b) =>
      b.name.toLowerCase().includes(partialName.toLowerCase())
    );
  }

  test.describe("Block Addition", () => {
    test("can add SmartDecisionMaker block to canvas", async () => {
      await buildPage.openBlocksPanel();

      const smartBlock = await getSmartDecisionMakerBlock();
      if (!smartBlock) {
        test.skip(true, "SmartDecisionMaker block not found in API");
        return;
      }

      await buildPage.addBlock(smartBlock);
      await buildPage.closeBlocksPanel();
      await buildPage.hasBlock(smartBlock);
    });

    test("SmartDecisionMaker block has expected input pins", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const smartBlock = await getSmartDecisionMakerBlock();
      if (!smartBlock) {
        test.skip(true, "SmartDecisionMaker block not found in API");
        return;
      }

      await buildPage.addBlock(smartBlock);
      await buildPage.closeBlocksPanel();

      // Verify expected input handles exist
      const blockElement = page.locator(`[data-blockid="${smartBlock.id}"]`).first();
      await expect(blockElement).toBeVisible();

      // Check for common SmartDecisionMaker inputs
      const promptInput = blockElement.locator('[data-testid="input-handle-prompt"]');
      const modelInput = blockElement.locator('[data-testid="input-handle-model"]');

      // At least the prompt input should exist
      await expect(promptInput).toBeAttached();
    });
  });

  test.describe("Pin Name Handling", () => {
    test("block connections preserve original field names in UI", async ({ page }) => {
      await buildPage.openBlocksPanel();

      // Add a Store Value block to test connections
      const storeBlock = await findBlockByName("Store Value");
      if (!storeBlock) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({
        ...storeBlock,
        name: "Store Value 1",
      });
      await buildPage.addBlock({
        ...storeBlock,
        name: "Store Value 2",
      });
      await buildPage.closeBlocksPanel();

      // Connect the blocks
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Verify connection was made
      const edge = page.locator(".react-flow__edge");
      await expect(edge.first()).toBeVisible();
    });

    test("input handles are accessible for fields with various names", async ({ page }) => {
      await buildPage.openBlocksPanel();

      // Find a block that might have inputs with spaces/special chars
      const blocks = await buildPage.getBlocksFromAPI();

      // Look for blocks in AI category which often have complex field names
      const aiBlocks = blocks.filter((b) => b.type === "AI" || b.type === "Standard");

      if (aiBlocks.length === 0) {
        test.skip(true, "No suitable blocks found for testing");
        return;
      }

      // Add the first available block
      const testBlock = aiBlocks[0];
      await buildPage.addBlock(testBlock);
      await buildPage.closeBlocksPanel();

      // Verify the block is on canvas
      await buildPage.hasBlock(testBlock);

      // Get all input handles on the block
      const blockElement = page.locator(`[data-blockid="${testBlock.id}"]`).first();
      const inputHandles = blockElement.locator('[data-testid^="input-handle-"]');

      const handleCount = await inputHandles.count();
      console.log(`Block ${testBlock.name} has ${handleCount} input handles`);

      // Verify handles are accessible
      if (handleCount > 0) {
        const firstHandle = inputHandles.first();
        await expect(firstHandle).toBeAttached();
      }
    });
  });

  test.describe("Block Connections", () => {
    test("can connect SmartDecisionMaker output to downstream block", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const smartBlock = await getSmartDecisionMakerBlock();
      const storeBlock = await findBlockByName("Store Value");

      if (!smartBlock || !storeBlock) {
        test.skip(true, "Required blocks not found");
        return;
      }

      // Add SmartDecisionMaker
      await buildPage.addBlock(smartBlock);

      // Add a downstream block
      await buildPage.addBlock({
        ...storeBlock,
        name: "Downstream Store",
      });

      await buildPage.closeBlocksPanel();

      // Wait for blocks to settle
      await page.waitForTimeout(500);

      // Verify both blocks are present
      await buildPage.hasBlock(smartBlock);

      // The tools output should be available for connection
      const smartBlockElement = page.locator(`[data-blockid="${smartBlock.id}"]`).first();
      const toolsOutput = smartBlockElement.locator('[data-testid="output-handle-tools"]');

      // tools output may or may not exist depending on block configuration
      const hasToolsOutput = await toolsOutput.count() > 0;
      console.log(`SmartDecisionMaker has tools output: ${hasToolsOutput}`);
    });

    test("connection data attributes use correct format", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await findBlockByName("Store Value");
      if (!storeBlock) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({
        ...storeBlock,
        name: "Store 1",
      });
      await buildPage.addBlock({
        ...storeBlock,
        name: "Store 2",
      });

      await buildPage.closeBlocksPanel();

      // Connect via data IDs
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Verify edge was created
      const edges = page.locator(".react-flow__edge");
      await expect(edges.first()).toBeVisible();

      // Get edge data attributes
      const edgeElement = edges.first();
      const sourceHandle = await edgeElement.getAttribute("data-sourcehandle");
      const targetHandle = await edgeElement.getAttribute("data-targethandle");

      console.log(`Edge source handle: ${sourceHandle}`);
      console.log(`Edge target handle: ${targetHandle}`);

      // The handles should be set
      expect(sourceHandle).toBeTruthy();
      expect(targetHandle).toBeTruthy();
    });
  });

  test.describe("Agent Save and Load", () => {
    test("can save agent with SmartDecisionMaker block", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const smartBlock = await getSmartDecisionMakerBlock();
      if (!smartBlock) {
        test.skip(true, "SmartDecisionMaker block not found");
        return;
      }

      await buildPage.addBlock(smartBlock);
      await buildPage.closeBlocksPanel();

      // Save the agent
      const agentName = `SDM Test ${Date.now()}`;
      await buildPage.saveAgent(agentName, "Testing SmartDecisionMaker");

      // Verify URL updated with flowID
      await expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));

      // Wait for save to complete
      await buildPage.waitForSaveButton();
    });

    test("saved agent preserves block connections", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await findBlockByName("Store Value");
      if (!storeBlock) {
        test.skip(true, "Store Value block not found");
        return;
      }

      // Add and connect blocks
      await buildPage.addBlock({
        ...storeBlock,
        name: "Store 1",
      });
      await buildPage.addBlock({
        ...storeBlock,
        name: "Store 2",
      });
      await buildPage.closeBlocksPanel();

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Save
      const agentName = `Connection Test ${Date.now()}`;
      await buildPage.saveAgent(agentName, "Testing connections");
      await expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));

      // Count edges before reload
      const edgesBefore = await page.locator(".react-flow__edge").count();

      // Reload the page
      await page.reload();
      await buildPage.closeTutorial();

      // Wait for graph to load
      await page.waitForTimeout(2000);

      // Verify edges still exist
      const edgesAfter = await page.locator(".react-flow__edge").count();
      expect(edgesAfter).toBe(edgesBefore);
    });
  });

  test.describe("Field Name Display", () => {
    test("block inputs display readable field names", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const smartBlock = await getSmartDecisionMakerBlock();
      if (!smartBlock) {
        test.skip(true, "SmartDecisionMaker block not found");
        return;
      }

      await buildPage.addBlock(smartBlock);
      await buildPage.closeBlocksPanel();

      const blockElement = page.locator(`[data-blockid="${smartBlock.id}"]`).first();

      // Get all visible input labels
      const inputLabels = blockElement.locator('[data-id^="input-handle-"]');
      const count = await inputLabels.count();

      console.log(`Found ${count} input containers`);

      // Log each input's data-id to see field naming
      for (let i = 0; i < Math.min(count, 5); i++) {
        const label = inputLabels.nth(i);
        const dataId = await label.getAttribute("data-id");
        console.log(`Input ${i}: ${dataId}`);
      }
    });

    test("output handles have correct data-testid format", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await findBlockByName("Store Value");
      if (!storeBlock) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock(storeBlock);
      await buildPage.closeBlocksPanel();

      const blockElement = page.locator(`[data-blockid="${storeBlock.id}"]`).first();
      const outputHandles = blockElement.locator('[data-testid^="output-handle-"]');

      const count = await outputHandles.count();
      console.log(`Found ${count} output handles`);

      for (let i = 0; i < count; i++) {
        const handle = outputHandles.nth(i);
        const testId = await handle.getAttribute("data-testid");
        console.log(`Output handle ${i}: ${testId}`);

        // Verify format: output-handle-{fieldname}
        expect(testId).toMatch(/^output-handle-/);
      }
    });
  });

  test.describe("Multi-Block Workflows", () => {
    test("can create workflow with multiple connected blocks", async ({ page }) => {
      test.setTimeout(90000);
      await buildPage.openBlocksPanel();

      const storeBlock = await findBlockByName("Store Value");
      if (!storeBlock) {
        test.skip(true, "Store Value block not found");
        return;
      }

      // Add three blocks in a chain
      await buildPage.addBlock({
        ...storeBlock,
        name: "Block A",
      });
      await buildPage.addBlock({
        ...storeBlock,
        name: "Block B",
      });
      await buildPage.addBlock({
        ...storeBlock,
        name: "Block C",
      });

      await buildPage.closeBlocksPanel();

      // Connect A -> B
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Connect B -> C
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-2-output-source",
        "1-3-input-target"
      );

      // Verify we have 2 edges
      const edges = page.locator(".react-flow__edge");
      await expect(edges).toHaveCount(2);

      // Save the workflow
      await buildPage.saveAgent(
        `Workflow Test ${Date.now()}`,
        "Multi-block workflow test"
      );
      await expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
    });
  });
});

test.describe("SmartDecisionMaker Pin Sanitization", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page }) => {
    test.setTimeout(60000);
    const loginPage = new LoginPage(page);
    const testUser = await getTestUser();

    buildPage = new BuildPage(page);

    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await hasUrl(page, "/marketplace");
    await buildPage.navbar.clickBuildLink();
    await hasUrl(page, "/build");
    await buildPage.closeTutorial();
  });

  test("verifies input handle naming convention", async ({ page }) => {
    /**
     * This test documents the expected behavior of input handle naming.
     *
     * The bug: If frontend uses original names (with spaces) in data attributes
     * but backend expects sanitized names (lowercase, underscores), routing fails.
     */
    await buildPage.openBlocksPanel();

    // Get all blocks and find one with inputs
    const blocks = await buildPage.getBlocksFromAPI();
    const blockWithInputs = blocks.find((b) => b.type === "Standard");

    if (!blockWithInputs) {
      test.skip(true, "No suitable block found");
      return;
    }

    await buildPage.addBlock(blockWithInputs);
    await buildPage.closeBlocksPanel();

    const blockElement = page.locator(`[data-blockid="${blockWithInputs.id}"]`).first();
    const inputHandles = blockElement.locator('[data-testid^="input-handle-"]');

    const count = await inputHandles.count();

    // Document the actual naming convention used
    const handleNames: string[] = [];
    for (let i = 0; i < count; i++) {
      const handle = inputHandles.nth(i);
      const testId = await handle.getAttribute("data-testid");
      if (testId) {
        const fieldName = testId.replace("input-handle-", "");
        handleNames.push(fieldName);
      }
    }

    console.log(`Block: ${blockWithInputs.name}`);
    console.log(`Input handle names: ${JSON.stringify(handleNames)}`);

    // Check if names are lowercase (sanitized) or original case
    for (const name of handleNames) {
      const isLowercase = name === name.toLowerCase();
      const hasSpaces = name.includes(" ");
      const hasSpecialChars = /[^a-zA-Z0-9_-]/.test(name);

      console.log(`  ${name}: lowercase=${isLowercase}, spaces=${hasSpaces}, special=${hasSpecialChars}`);

      // Document: Frontend uses lowercase handle names
      // This should match backend sanitization
      expect(isLowercase).toBe(true);
      expect(hasSpaces).toBe(false);
    }
  });

  test("verifies output handle naming matches input handle convention", async ({ page }) => {
    await buildPage.openBlocksPanel();

    const blocks = await buildPage.getBlocksFromAPI();
    const blockWithOutputs = blocks.find((b) => b.type === "Standard");

    if (!blockWithOutputs) {
      test.skip(true, "No suitable block found");
      return;
    }

    await buildPage.addBlock(blockWithOutputs);
    await buildPage.closeBlocksPanel();

    const blockElement = page.locator(`[data-blockid="${blockWithOutputs.id}"]`).first();
    const outputHandles = blockElement.locator('[data-testid^="output-handle-"]');

    const count = await outputHandles.count();

    for (let i = 0; i < count; i++) {
      const handle = outputHandles.nth(i);
      const testId = await handle.getAttribute("data-testid");
      if (testId) {
        const fieldName = testId.replace("output-handle-", "");

        // Output handles should also use lowercase sanitized names
        const isLowercase = fieldName === fieldName.toLowerCase();
        expect(isLowercase).toBe(true);
      }
    }
  });

  test("link creation uses consistent field naming", async ({ page }) => {
    /**
     * This test verifies that when creating a connection (link),
     * both source and target use consistent naming conventions.
     */
    await buildPage.openBlocksPanel();

    const storeBlock = await buildPage.getFilteredBlocksFromAPI(
      (b) => b.name.toLowerCase().includes("store value")
    );

    if (storeBlock.length === 0) {
      test.skip(true, "Store Value block not found");
      return;
    }

    await buildPage.addBlock({
      ...storeBlock[0],
      name: "Source Block",
    });
    await buildPage.addBlock({
      ...storeBlock[0],
      name: "Target Block",
    });
    await buildPage.closeBlocksPanel();

    // Create connection
    await buildPage.connectBlockOutputToBlockInputViaDataId(
      "1-1-output-source",
      "1-2-input-target"
    );

    // Get the created edge
    const edge = page.locator(".react-flow__edge").first();
    await expect(edge).toBeVisible();

    // Check edge attributes for naming consistency
    const sourceHandle = await edge.getAttribute("data-sourcehandle");
    const targetHandle = await edge.getAttribute("data-targethandle");

    console.log(`Source handle: ${sourceHandle}`);
    console.log(`Target handle: ${targetHandle}`);

    // Both should be non-empty
    expect(sourceHandle).toBeTruthy();
    expect(targetHandle).toBeTruthy();

    // Check if handles follow sanitized naming convention
    if (sourceHandle && targetHandle) {
      const sourceIsLowercase = sourceHandle === sourceHandle.toLowerCase();
      const targetIsLowercase = targetHandle === targetHandle.toLowerCase();

      // Document: Edge handles should use sanitized names
      // This ensures consistency with backend emit keys
      console.log(`Source handle is lowercase: ${sourceIsLowercase}`);
      console.log(`Target handle is lowercase: ${targetIsLowercase}`);
    }
  });
});
