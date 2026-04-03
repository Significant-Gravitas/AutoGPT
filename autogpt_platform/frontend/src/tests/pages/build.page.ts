import { expect, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

export class BuildPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // --- Navigation ---

  async goto(): Promise<void> {
    await this.page.goto("/build");
    await this.page.waitForLoadState("domcontentloaded");
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
      await this.page
        .locator(".react-flow")
        .waitFor({ state: "visible", timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  async closeTutorial(): Promise<void> {
    try {
      await this.page
        .getByRole("button", { name: "Skip Tutorial", exact: true })
        .click({ timeout: 3000 });
    } catch {
      // Tutorial not shown or already dismissed
    }
  }

  // --- Block Menu ---

  async openBlocksPanel(): Promise<void> {
    const popoverContent = this.page.locator(
      '[data-id="blocks-control-popover-content"]',
    );
    if (!(await popoverContent.isVisible())) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
      await popoverContent.waitFor({ state: "visible", timeout: 5000 });
    }
  }

  async closeBlocksPanel(): Promise<void> {
    const popoverContent = this.page.locator(
      '[data-id="blocks-control-popover-content"]',
    );
    if (await popoverContent.isVisible()) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
      await popoverContent.waitFor({ state: "hidden", timeout: 5000 });
    }
  }

  async searchBlock(searchTerm: string): Promise<void> {
    const searchInput = this.page.locator(
      '[data-id="blocks-control-search-bar"] input[type="text"]',
    );
    await searchInput.clear();
    await searchInput.fill(searchTerm);
    await this.page.waitForTimeout(300);
  }

  private getBlockCardByName(name: string): Locator {
    const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const exactName = new RegExp(`^\\s*${escapedName}\\s*$`, "i");
    return this.page
      .locator('[data-id^="block-card-"]')
      .filter({ has: this.page.locator("span", { hasText: exactName }) })
      .first();
  }

  async addBlockByClick(searchTerm: string): Promise<void> {
    await this.openBlocksPanel();
    await this.searchBlock(searchTerm);

    // Wait for any search results to appear
    const anyCard = this.page.locator('[data-id^="block-card-"]').first();
    await anyCard.waitFor({ state: "visible", timeout: 10000 });

    // Click the card matching the search term name
    const blockCard = this.getBlockCardByName(searchTerm);
    await blockCard.waitFor({ state: "visible", timeout: 5000 });
    await blockCard.click();

    // Close the panel so it doesn't overlay the canvas
    await this.closeBlocksPanel();
  }

  async dragBlockToCanvas(searchTerm: string): Promise<void> {
    await this.openBlocksPanel();
    await this.searchBlock(searchTerm);

    const anyCard = this.page.locator('[data-id^="block-card-"]').first();
    await anyCard.waitFor({ state: "visible", timeout: 10000 });

    const blockCard = this.getBlockCardByName(searchTerm);
    await blockCard.waitFor({ state: "visible", timeout: 5000 });

    const canvas = this.page.locator(".react-flow__pane").first();
    await blockCard.dragTo(canvas);
  }

  // --- Nodes on Canvas ---

  getNodeLocator(index?: number): Locator {
    const locator = this.page.locator('[data-id^="custom-node-"]');
    return index !== undefined ? locator.nth(index) : locator;
  }

  async getNodeCount(): Promise<number> {
    return await this.getNodeLocator().count();
  }

  async waitForNodeOnCanvas(expectedCount?: number): Promise<void> {
    if (expectedCount !== undefined) {
      await expect(this.getNodeLocator()).toHaveCount(expectedCount, {
        timeout: 10000,
      });
    } else {
      await this.getNodeLocator()
        .first()
        .waitFor({ state: "visible", timeout: 10000 });
    }
  }

  async selectNode(index: number = 0): Promise<void> {
    const node = this.getNodeLocator(index);
    await node.click();
  }

  async selectAllNodes(): Promise<void> {
    await this.page.locator(".react-flow__pane").first().click();
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+a" : "Control+a");
  }

  async deleteSelectedNodes(): Promise<void> {
    await this.page.keyboard.press("Backspace");
  }

  // --- Connections (Edges) ---

  async connectNodes(
    sourceNodeIndex: number,
    targetNodeIndex: number,
  ): Promise<void> {
    // Get the node wrapper elements to scope handle search
    const sourceNode = this.getNodeLocator(sourceNodeIndex);
    const targetNode = this.getNodeLocator(targetNodeIndex);

    // ReactFlow renders Handle components as .react-flow__handle elements
    // Output handles have class .react-flow__handle-right (Position.Right)
    // Input handles have class .react-flow__handle-left (Position.Left)
    const sourceHandle = sourceNode
      .locator(".react-flow__handle-right")
      .first();
    const targetHandle = targetNode.locator(".react-flow__handle-left").first();

    // Get precise center coordinates using evaluate to avoid CSS transform issues
    const getHandleCenter = async (locator: Locator) => {
      const el = await locator.elementHandle();
      if (!el) throw new Error("Handle element not found");
      const rect = await el.evaluate((node) => {
        const r = node.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      });
      return rect;
    };

    const source = await getHandleCenter(sourceHandle);
    const target = await getHandleCenter(targetHandle);

    // ReactFlow requires a proper drag sequence with intermediate moves
    await this.page.mouse.move(source.x, source.y);
    await this.page.mouse.down();
    // Move in steps to trigger ReactFlow's connection detection
    const steps = 20;
    for (let i = 1; i <= steps; i++) {
      const ratio = i / steps;
      await this.page.mouse.move(
        source.x + (target.x - source.x) * ratio,
        source.y + (target.y - source.y) * ratio,
      );
    }
    await this.page.mouse.up();
  }

  async getEdgeCount(): Promise<number> {
    return await this.page.locator(".react-flow__edge").count();
  }

  // --- Save ---

  async saveAgent(
    name: string = "Test Agent",
    description: string = "",
  ): Promise<void> {
    await this.page.getByTestId("save-control-save-button").click();

    const nameInput = this.page.getByTestId("save-control-name-input");
    await nameInput.waitFor({ state: "visible", timeout: 5000 });
    await nameInput.fill(name);

    if (description) {
      await this.page
        .getByTestId("save-control-description-input")
        .fill(description);
    }

    await this.page.getByTestId("save-control-save-agent-button").click();
  }

  async waitForSaveComplete(): Promise<void> {
    await expect(this.page).toHaveURL(/flowID=/, { timeout: 15000 });
  }

  async waitForSaveButton(): Promise<void> {
    await this.page.waitForSelector(
      '[data-testid="save-control-save-button"]:not([disabled])',
      { timeout: 10000 },
    );
  }

  // --- Run ---

  async isRunButtonEnabled(): Promise<boolean> {
    const runButton = this.page.locator('[data-id="run-graph-button"]');
    return await runButton.isEnabled();
  }

  async clickRunButton(): Promise<void> {
    const runButton = this.page.locator('[data-id="run-graph-button"]');
    await runButton.click();
  }

  // --- Undo / Redo ---

  async isUndoEnabled(): Promise<boolean> {
    const btn = this.page.locator('[data-id="undo-button"]');
    return !(await btn.isDisabled());
  }

  async isRedoEnabled(): Promise<boolean> {
    const btn = this.page.locator('[data-id="redo-button"]');
    return !(await btn.isDisabled());
  }

  async clickUndo(): Promise<void> {
    await this.page.locator('[data-id="undo-button"]').click();
  }

  async clickRedo(): Promise<void> {
    await this.page.locator('[data-id="redo-button"]').click();
  }

  // --- Copy / Paste ---

  async copyViaKeyboard(): Promise<void> {
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+c" : "Control+c");
  }

  async pasteViaKeyboard(): Promise<void> {
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+v" : "Control+v");
  }

  // --- Helpers ---

  async fillBlockInputByPlaceholder(
    placeholder: string,
    value: string,
    nodeIndex: number = 0,
  ): Promise<void> {
    const node = this.getNodeLocator(nodeIndex);
    const input = node.getByPlaceholder(placeholder);
    await input.fill(value);
  }

  async clickCanvas(): Promise<void> {
    const pane = this.page.locator(".react-flow__pane").first();
    const box = await pane.boundingBox();
    if (box) {
      // Click in the center of the canvas to avoid sidebar/toolbar overlaps
      await pane.click({
        position: { x: box.width / 2, y: box.height / 2 },
      });
    } else {
      await pane.click();
    }
  }

  getPlaywrightPage(): Page {
    return this.page;
  }

  async createDummyAgent(): Promise<void> {
    await this.closeTutorial();
    await this.addBlockByClick("Add to Dictionary");
    await this.waitForNodeOnCanvas(1);
    await this.saveAgent("Test Agent", "Test Description");
    await this.waitForSaveComplete();
  }
}
