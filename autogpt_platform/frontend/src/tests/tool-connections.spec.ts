/**
 * E2E tests for tool connections and routing in the graph builder.
 *
 * These tests focus on the connection behavior between blocks,
 * particularly around the SmartDecisionMaker tools output routing.
 *
 * Key scenarios tested:
 * 1. Connection data attribute formats
 * 2. Handle naming conventions
 * 3. Edge creation with various field name formats
 * 4. Link persistence after save/reload
 */

import test, { expect } from "@playwright/test";
import { BuildPage, Block } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl } from "./utils/assertion";
import { getTestUser } from "./utils/auth";

test.describe("Tool Connections", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page }) => {
    test.setTimeout(45000);
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

  test.describe("Connection Data Attributes", () => {
    test("edge source and target handles are set correctly", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({ ...storeBlock[0], name: "Source" });
      await buildPage.addBlock({ ...storeBlock[0], name: "Target" });
      await buildPage.closeBlocksPanel();

      // Connect blocks
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Verify edge exists and has correct attributes
      const edge = page.locator(".react-flow__edge").first();
      await expect(edge).toBeVisible();

      // Get all relevant edge attributes
      const attributes = await edge.evaluate((el) => ({
        source: el.getAttribute("data-source"),
        target: el.getAttribute("data-target"),
        sourceHandle: el.getAttribute("data-sourcehandle"),
        targetHandle: el.getAttribute("data-targethandle"),
        id: el.getAttribute("id"),
      }));

      console.log("Edge attributes:", JSON.stringify(attributes, null, 2));

      // Source and target should be node IDs
      expect(attributes.source).toBeTruthy();
      expect(attributes.target).toBeTruthy();

      // Handles should reference field names
      expect(attributes.sourceHandle).toBeTruthy();
      expect(attributes.targetHandle).toBeTruthy();
    });

    test("edge ID follows expected format", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({ ...storeBlock[0], name: "A" });
      await buildPage.addBlock({ ...storeBlock[0], name: "B" });
      await buildPage.closeBlocksPanel();

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      const edge = page.locator(".react-flow__edge").first();
      const edgeId = await edge.getAttribute("id");

      console.log(`Edge ID: ${edgeId}`);

      // Edge ID typically contains source-target info
      expect(edgeId).toBeTruthy();
      // Format: reactflow__edge-{source}{sourceHandle}-{target}{targetHandle}
      expect(edgeId).toContain("reactflow__edge");
    });
  });

  test.describe("Handle Naming Consistency", () => {
    test("all input handles use lowercase naming", async ({ page }) => {
      await buildPage.openBlocksPanel();

      // Get multiple blocks to test variety
      const blocks = await buildPage.getBlocksFromAPI();
      const testBlocks = blocks.slice(0, 3).filter((b) => b.type !== "Agent");

      if (testBlocks.length === 0) {
        test.skip(true, "No suitable blocks found");
        return;
      }

      for (const block of testBlocks) {
        await buildPage.addBlock(block);
      }
      await buildPage.closeBlocksPanel();

      // Check all input handles across all blocks
      const allInputHandles = page.locator('[data-testid^="input-handle-"]');
      const count = await allInputHandles.count();

      let uppercaseFound = false;
      let spacesFound = false;

      for (let i = 0; i < count; i++) {
        const handle = allInputHandles.nth(i);
        const testId = await handle.getAttribute("data-testid");

        if (testId) {
          const fieldName = testId.replace("input-handle-", "");

          if (fieldName !== fieldName.toLowerCase()) {
            console.log(`Non-lowercase input handle found: ${fieldName}`);
            uppercaseFound = true;
          }

          if (fieldName.includes(" ")) {
            console.log(`Input handle with spaces found: ${fieldName}`);
            spacesFound = true;
          }
        }
      }

      // Document: Frontend should use lowercase sanitized names
      // If this fails, there's an inconsistency that could cause routing issues
      expect(uppercaseFound).toBe(false);
      expect(spacesFound).toBe(false);
    });

    test("all output handles use lowercase naming", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const blocks = await buildPage.getBlocksFromAPI();
      const testBlocks = blocks.slice(0, 3).filter((b) => b.type !== "Agent");

      if (testBlocks.length === 0) {
        test.skip(true, "No suitable blocks found");
        return;
      }

      for (const block of testBlocks) {
        await buildPage.addBlock(block);
      }
      await buildPage.closeBlocksPanel();

      const allOutputHandles = page.locator('[data-testid^="output-handle-"]');
      const count = await allOutputHandles.count();

      let uppercaseFound = false;
      let spacesFound = false;

      for (let i = 0; i < count; i++) {
        const handle = allOutputHandles.nth(i);
        const testId = await handle.getAttribute("data-testid");

        if (testId) {
          const fieldName = testId.replace("output-handle-", "");

          if (fieldName !== fieldName.toLowerCase()) {
            uppercaseFound = true;
            console.log(`Non-lowercase output handle: ${fieldName}`);
          }

          if (fieldName.includes(" ")) {
            spacesFound = true;
            console.log(`Output handle with spaces: ${fieldName}`);
          }
        }
      }

      expect(uppercaseFound).toBe(false);
      expect(spacesFound).toBe(false);
    });
  });

  test.describe("Connection Persistence", () => {
    test("connections survive page reload", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({ ...storeBlock[0], name: "Persist A" });
      await buildPage.addBlock({ ...storeBlock[0], name: "Persist B" });
      await buildPage.closeBlocksPanel();

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Save the agent
      await buildPage.saveAgent(
        `Persist Test ${Date.now()}`,
        "Testing connection persistence"
      );
      await expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
      await buildPage.waitForSaveButton();

      // Get current URL
      const url = page.url();

      // Reload
      await page.reload();
      await buildPage.closeTutorial();
      await page.waitForTimeout(2000);

      // Verify edge still exists
      const edge = page.locator(".react-flow__edge").first();
      await expect(edge).toBeVisible();

      // Verify same URL
      expect(page.url()).toBe(url);
    });

    test("connection attributes preserved after save", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({ ...storeBlock[0], name: "Attr A" });
      await buildPage.addBlock({ ...storeBlock[0], name: "Attr B" });
      await buildPage.closeBlocksPanel();

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      // Get attributes before save
      const edgeBefore = page.locator(".react-flow__edge").first();
      const attrsBefore = await edgeBefore.evaluate((el) => ({
        sourceHandle: el.getAttribute("data-sourcehandle"),
        targetHandle: el.getAttribute("data-targethandle"),
      }));

      // Save
      await buildPage.saveAgent(`Attr Test ${Date.now()}`, "Testing attributes");
      await expect(page).toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));
      await buildPage.waitForSaveButton();

      // Reload
      await page.reload();
      await buildPage.closeTutorial();
      await page.waitForTimeout(2000);

      // Get attributes after reload
      const edgeAfter = page.locator(".react-flow__edge").first();
      await expect(edgeAfter).toBeVisible();

      const attrsAfter = await edgeAfter.evaluate((el) => ({
        sourceHandle: el.getAttribute("data-sourcehandle"),
        targetHandle: el.getAttribute("data-targethandle"),
      }));

      console.log("Before save:", attrsBefore);
      console.log("After reload:", attrsAfter);

      // Handle names should be preserved
      expect(attrsAfter.sourceHandle).toBe(attrsBefore.sourceHandle);
      expect(attrsAfter.targetHandle).toBe(attrsBefore.targetHandle);
    });
  });

  test.describe("Multiple Connections", () => {
    test("can create multiple connections from single output", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      // Add one source and two targets
      await buildPage.addBlock({ ...storeBlock[0], name: "Multi Source" });
      await buildPage.addBlock({ ...storeBlock[0], name: "Target 1" });
      await buildPage.addBlock({ ...storeBlock[0], name: "Target 2" });
      await buildPage.closeBlocksPanel();

      // Connect source to both targets
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-3-input-target"
      );

      // Should have 2 edges
      const edges = page.locator(".react-flow__edge");
      await expect(edges).toHaveCount(2);
    });

    test("each connection has unique edge ID", async ({ page }) => {
      await buildPage.openBlocksPanel();

      const storeBlock = await buildPage.getFilteredBlocksFromAPI(
        (b) => b.name.toLowerCase().includes("store value")
      );

      if (storeBlock.length === 0) {
        test.skip(true, "Store Value block not found");
        return;
      }

      await buildPage.addBlock({ ...storeBlock[0], name: "ID Source" });
      await buildPage.addBlock({ ...storeBlock[0], name: "ID Target 1" });
      await buildPage.addBlock({ ...storeBlock[0], name: "ID Target 2" });
      await buildPage.closeBlocksPanel();

      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-2-input-target"
      );
      await buildPage.connectBlockOutputToBlockInputViaDataId(
        "1-1-output-source",
        "1-3-input-target"
      );

      const edges = page.locator(".react-flow__edge");
      const edgeIds: string[] = [];

      const count = await edges.count();
      for (let i = 0; i < count; i++) {
        const edge = edges.nth(i);
        const id = await edge.getAttribute("id");
        if (id) edgeIds.push(id);
      }

      console.log("Edge IDs:", edgeIds);

      // All IDs should be unique
      const uniqueIds = new Set(edgeIds);
      expect(uniqueIds.size).toBe(edgeIds.length);
    });
  });
});

test.describe("Tool Output Pin Format", () => {
  let buildPage: BuildPage;

  test.beforeEach(async ({ page }) => {
    test.setTimeout(45000);
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

  test("documents tool output pin naming format", async ({ page }) => {
    /**
     * This test documents the expected format for tool output pins
     * which is critical for routing to work correctly.
     *
     * Expected format: tools_^_{sink_node_id}_~_{sanitized_field_name}
     *
     * The bug occurs when:
     * - Frontend creates link with: tools_^_{node}_~_Max Keyword Difficulty
     * - Backend emits with: tools_^_{node}_~_max_keyword_difficulty
     */
    await buildPage.openBlocksPanel();

    // Look for SmartDecisionMaker or any AI block
    const blocks = await buildPage.getBlocksFromAPI();
    const aiBlock = blocks.find(
      (b) =>
        b.type === "AI" ||
        b.name.toLowerCase().includes("smart") ||
        b.name.toLowerCase().includes("decision")
    );

    if (!aiBlock) {
      console.log("No AI block found, documenting expected format:");
      console.log("Tool pin format: tools_^_{sink_node_id}_~_{sanitized_field_name}");
      console.log("Example: tools_^_abc-123_~_max_keyword_difficulty");
      test.skip(true, "No AI block available for testing");
      return;
    }

    await buildPage.addBlock(aiBlock);
    await buildPage.closeBlocksPanel();

    const blockElement = page.locator(`[data-blockid="${aiBlock.id}"]`).first();

    // Get tools output handle if it exists
    const toolsOutput = blockElement.locator('[data-testid="output-handle-tools"]');
    const hasToolsOutput = (await toolsOutput.count()) > 0;

    if (hasToolsOutput) {
      console.log("Tools output pin found");

      // Document the expected behavior
      // When this pin is connected, the link should use sanitized names
    } else {
      console.log("No tools output pin on this block");
    }

    // Document expected format regardless
    console.log("\nExpected tool pin format for SmartDecisionMaker:");
    console.log("  Source: tools_^_{sink_node_id}_~_{sanitized_field_name}");
    console.log("  Example sink_pin_name: max_keyword_difficulty (NOT 'Max Keyword Difficulty')");
  });
});
