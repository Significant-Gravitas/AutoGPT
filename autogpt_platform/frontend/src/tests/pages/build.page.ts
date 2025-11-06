import { expect, Locator, Page } from "@playwright/test";
import { Block as APIBlock } from "../../lib/autogpt-server-api/types";
import { beautifyString } from "../../lib/utils";
import { isVisible } from "../utils/assertion";
import { BasePage } from "./base.page";

export interface Block {
  id: string;
  name: string;
  description: string;
  type: string;
}

export class BuildPage extends BasePage {
  private cachedBlocks: Record<string, Block> = {};

  constructor(page: Page) {
    super(page);
  }

  private getDisplayName(blockName: string): string {
    return beautifyString(blockName).replace(/ Block$/, "");
  }

  async closeTutorial(): Promise<void> {
    console.log(`closing tutorial`);
    try {
      await this.page
        .getByRole("button", { name: "Skip Tutorial", exact: true })
        .click();
    } catch (error) {
      console.info("Error closing tutorial:", error);
    }
  }

  async openBlocksPanel(): Promise<void> {
    const isPanelOpen = await this.page
      .getByTestId("blocks-control-blocks-label")
      .isVisible();

    if (!isPanelOpen) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
    }
  }

  async closeBlocksPanel(): Promise<void> {
    await this.page.getByTestId("profile-popout-menu-trigger").click();
  }

  async saveAgent(
    name: string = "Test Agent",
    description: string = "",
  ): Promise<void> {
    console.log(`💾 Saving agent '${name}' with description '${description}'`);
    await this.page.getByTestId("blocks-control-save-button").click();
    await this.page.getByTestId("save-control-name-input").fill(name);
    await this.page
      .getByTestId("save-control-description-input")
      .fill(description);
    await this.page.getByTestId("save-control-save-agent-button").click();
  }

  async getBlocksFromAPI(): Promise<Block[]> {
    if (Object.keys(this.cachedBlocks).length > 0) {
      return Object.values(this.cachedBlocks);
    }

    console.log(`Getting blocks from API request`);

    // Make direct API request using the page's request context
    const response = await this.page.request.get(
      "http://localhost:3000/api/proxy/api/blocks",
    );
    const apiBlocks: APIBlock[] = await response.json();

    console.log(`Found ${apiBlocks.length} blocks from API`);

    // Convert API blocks to test Block format
    const blocks = apiBlocks.map((block) => ({
      id: block.id,
      name: block.name,
      description: block.description,
      type: block.uiType,
    }));

    this.cachedBlocks = blocks.reduce(
      (acc, block) => {
        acc[block.id] = block;
        return acc;
      },
      {} as Record<string, Block>,
    );
    return blocks;
  }

  async getFilteredBlocksFromAPI(
    filterFn: (block: Block) => boolean,
  ): Promise<Block[]> {
    console.log(`Getting filtered blocks from API`);
    const blocks = await this.getBlocksFromAPI();
    return blocks.filter(filterFn);
  }

  async addBlock(block: Block): Promise<void> {
    console.log(`Adding block ${block.name} (${block.id}) to agent`);

    await this.openBlocksPanel();

    const searchInput = this.page.locator(
      '[data-id="blocks-control-search-input"]',
    );

    const displayName = this.getDisplayName(block.name);
    await searchInput.clear();
    await searchInput.fill(displayName);

    const blockCard = this.page.getByTestId(`block-name-${block.id}`);

    try {
      // Wait for the block card to be visible with a reasonable timeout
      await blockCard.waitFor({ state: "visible", timeout: 10000 });
      await blockCard.click();
      const blockInEditor = this.page.getByTestId(block.id).first();
      expect(blockInEditor).toBeAttached();
    } catch (error) {
      console.log(
        `❌ ❌  Block ${block.name} (display: ${displayName}) returned from the API but not found in block list`,
      );
      console.log(`Error: ${error}`);
    }
  }

  async hasBlock(block: Block) {
    const blockInEditor = this.page.getByTestId(block.id).first();
    await blockInEditor.isVisible();
  }

  async getBlockInputs(blockId: string): Promise<string[]> {
    console.log(`Getting block ${blockId} inputs`);
    try {
      const node = this.page.locator(`[data-blockid="${blockId}"]`).first();
      const inputsData = await node.getAttribute("data-inputs");
      return inputsData ? JSON.parse(inputsData) : [];
    } catch (error) {
      console.error("Error getting block inputs:", error);
      return [];
    }
  }

  async selectBlockCategory(category: string): Promise<void> {
    console.log(`Selecting block category: ${category}`);
    await this.page.getByText(category, { exact: true }).click();
    // Wait for the blocks to load after category selection
    await this.page.waitForTimeout(3000);
  }

  async getBlocksForCategory(category: string): Promise<Block[]> {
    console.log(`Getting blocks for category: ${category}`);

    // Clear any existing search to ensure we see all blocks in the category
    const searchInput = this.page.locator(
      '[data-id="blocks-control-search-input"]',
    );
    await searchInput.clear();

    // Wait for search to clear
    await this.page.waitForTimeout(300);

    // Select the category first
    await this.selectBlockCategory(category);

    try {
      const blockFinder = this.page.locator('[data-id^="block-card-"]');
      await blockFinder.first().waitFor();
      const blocks = await blockFinder.all();

      console.log(`found ${blocks.length} blocks in category ${category}`);

      const results = await Promise.all(
        blocks.map(async (block) => {
          try {
            const fullId = (await block.getAttribute("data-id")) || "";
            const id = fullId.replace("block-card-", "");
            const nameElement = block.locator('[data-testid^="block-name-"]');
            const descriptionElement = block.locator(
              '[data-testid^="block-description-"]',
            );

            const name = (await nameElement.textContent()) || "";
            const description = (await descriptionElement.textContent()) || "";
            const type = (await nameElement.getAttribute("data-type")) || "";

            return {
              id,
              name: name.trim(),
              type: type.trim(),
              description: description.trim(),
            };
          } catch (elementError) {
            console.error("Error processing block:", elementError);
            return null;
          }
        }),
      );

      // Filter out any null results from errors
      return results.filter((block): block is Block => block !== null);
    } catch (error) {
      console.error(`Error getting blocks for category ${category}:`, error);
      return [];
    }
  }

  async _buildBlockSelector(blockId: string, dataId?: string): Promise<string> {
    const selector = dataId
      ? `[data-id="${dataId}"] [data-blockid="${blockId}"]`
      : `[data-blockid="${blockId}"]`;
    return selector;
  }

  private async moveBlockToViewportPosition(
    blockSelector: string,
    options: { xRatio?: number; yRatio?: number } = {},
  ): Promise<void> {
    const { xRatio = 0.5, yRatio = 0.5 } = options;
    const blockLocator = this.page.locator(blockSelector).first();

    await blockLocator.waitFor({ state: "visible" });

    const boundingBox = await blockLocator.boundingBox();
    const viewport = this.page.viewportSize();

    if (!boundingBox || !viewport) {
      return;
    }

    const currentX = boundingBox.x + boundingBox.width / 2;
    const currentY = boundingBox.y + boundingBox.height / 2;

    const targetX = viewport.width * xRatio;
    const targetY = viewport.height * yRatio;

    const distance = Math.hypot(targetX - currentX, targetY - currentY);

    if (distance < 5) {
      return;
    }

    await this.page.mouse.move(currentX, currentY);
    await this.page.mouse.down();
    await this.page.mouse.move(targetX, targetY, { steps: 15 });
    await this.page.mouse.up();
    await this.page.waitForTimeout(200);
  }

  async getBlockById(blockId: string, dataId?: string): Promise<Locator> {
    console.log(`getting block ${blockId} with dataId ${dataId}`);
    return this.page.locator(await this._buildBlockSelector(blockId, dataId));
  }

  // dataId is optional, if provided, it will start the search with that container, otherwise it will start with the blockId
  // this is useful if you have multiple blocks with the same id, but different dataIds which you should have when adding a block to the graph.
  // Do note that once you run an agent, the dataId will change, so you will need to update the tests to use the new dataId or not use the same block in tests that run an agent
  async fillBlockInputByPlaceholder(
    blockId: string,
    placeholder: string,
    value: string,
    dataId?: string,
  ): Promise<void> {
    console.log(
      `filling block input ${placeholder} with value ${value} of block ${blockId}`,
    );
    const block = await this.getBlockById(blockId, dataId);
    const input = block.getByPlaceholder(placeholder);
    await input.fill(value);
  }

  async selectBlockInputValue(
    blockId: string,
    inputName: string,
    value: string,
    dataId?: string,
  ): Promise<void> {
    console.log(
      `selecting value ${value} for input ${inputName} of block ${blockId}`,
    );
    // First get the button that opens the dropdown
    const baseSelector = await this._buildBlockSelector(blockId, dataId);

    // Find the combobox button within the input handle container
    const comboboxSelector = `${baseSelector} [data-id="input-handle-${inputName.toLowerCase()}"] button[role="combobox"]`;

    try {
      // Click the combobox to open it
      await this.page.click(comboboxSelector);

      // Wait a moment for the dropdown to open
      await this.page.waitForTimeout(100);

      // Select the option from the dropdown
      // The actual selector for the option might need adjustment based on the dropdown structure
      await this.page.getByRole("option", { name: value }).click();
    } catch (error) {
      console.error(
        `Error selecting value "${value}" for input "${inputName}":`,
        error,
      );
      throw error;
    }
  }

  async fillBlockInputByLabel(
    blockId: string,
    label: string,
    value: string,
  ): Promise<void> {
    console.log(`filling block input ${label} with value ${value}`);
    const block = await this.getBlockById(blockId);
    const input = block.getByLabel(label);
    await input.fill(value);
  }

  async connectBlockOutputToBlockInputViaDataId(
    blockOutputId: string,
    blockInputId: string,
  ): Promise<void> {
    console.log(
      `connecting block output ${blockOutputId} to block input ${blockInputId}`,
    );
    try {
      // Locate the output element
      const outputElement = this.page.locator(`[data-id="${blockOutputId}"]`);
      // Locate the input element
      const inputElement = this.page.locator(`[data-id="${blockInputId}"]`);

      await outputElement.dragTo(inputElement);
    } catch (error) {
      console.error("Error connecting block output to input:", error);
    }
  }

  async connectBlockOutputToBlockInputViaName(
    startBlockId: string,
    startBlockOutputName: string,
    endBlockId: string,
    endBlockInputName: string,
    startDataId?: string,
    endDataId?: string,
  ): Promise<void> {
    console.log(
      `connecting block output ${startBlockOutputName} of block ${startBlockId} to block input ${endBlockInputName} of block ${endBlockId}`,
    );

    const startBlockBase = await this._buildBlockSelector(
      startBlockId,
      startDataId,
    );

    const endBlockBase = await this._buildBlockSelector(endBlockId, endDataId);

    await this.moveBlockToViewportPosition(startBlockBase, { xRatio: 0.35 });
    await this.moveBlockToViewportPosition(endBlockBase, { xRatio: 0.65 });

    const startBlockOutputSelector = `${startBlockBase} [data-testid="output-handle-${startBlockOutputName.toLowerCase()}"]`;
    const endBlockInputSelector = `${endBlockBase} [data-testid="input-handle-${endBlockInputName.toLowerCase()}"]`;

    console.log("Start block selector:", startBlockOutputSelector);
    console.log("End block selector:", endBlockInputSelector);

    const startElement = this.page.locator(startBlockOutputSelector);
    const endElement = this.page.locator(endBlockInputSelector);

    await startElement.scrollIntoViewIfNeeded();
    await this.page.waitForTimeout(200);

    await endElement.scrollIntoViewIfNeeded();
    await this.page.waitForTimeout(200);

    await startElement.dragTo(endElement);
  }

  async isLoaded(): Promise<boolean> {
    console.log(`checking if build page is loaded`);
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  async isRunButtonEnabled(): Promise<boolean> {
    console.log(`checking if run button is enabled`);
    const runButton = this.page.getByTestId("primary-action-run-agent");
    return await runButton.isEnabled();
  }

  async runAgent(): Promise<void> {
    console.log(`clicking run button`);
    const runButton = this.page.getByTestId("primary-action-run-agent");
    await runButton.click();
    await this.page.waitForTimeout(1000);
    await runButton.click();
  }

  async fillRunDialog(inputs: Record<string, string>): Promise<void> {
    console.log(`filling run dialog`);
    for (const [key, value] of Object.entries(inputs)) {
      await this.page.getByTestId(`agent-input-${key}`).fill(value);
    }
  }
  async clickRunDialogRunButton(): Promise<void> {
    console.log(`clicking run button`);
    await this.page.getByTestId("agent-run-button").click();
  }

  async waitForCompletionBadge(): Promise<void> {
    console.log(`waiting for completion badge`);
    await this.page.waitForSelector(
      '[data-id^="badge-"][data-id$="-COMPLETED"]',
    );
  }

  async waitForSaveButton(): Promise<void> {
    console.log(`waiting for save button`);
    await this.page.waitForSelector(
      '[data-testid="blocks-control-save-button"]:not([disabled])',
    );
  }

  async isCompletionBadgeVisible(): Promise<boolean> {
    console.log(`checking for completion badge`);
    const completionBadge = this.page
      .locator('[data-id^="badge-"][data-id$="-COMPLETED"]')
      .first();
    return await completionBadge.isVisible();
  }

  async waitForVersionField(): Promise<void> {
    console.log(`waiting for version field`);

    // wait for the url to have the flowID
    await this.page.waitForSelector(
      '[data-testid="save-control-version-output"]',
    );
  }

  async getDictionaryBlockDetails(): Promise<Block> {
    return {
      id: "dummy-id-1",
      name: "Add to Dictionary",
      description: "Add to Dictionary",
      type: "Standard",
    };
  }

  async getCalculatorBlockDetails(): Promise<Block> {
    return {
      id: "dummy-id-2",
      name: "Calculator",
      description: "Calculator",
      type: "Standard",
    };
  }

  async waitForSaveDialogClose(): Promise<void> {
    console.log(`waiting for save dialog to close`);

    await this.page.waitForSelector(
      '[data-id="save-control-popover-content"]',
      { state: "hidden" },
    );
  }

  async getGithubTriggerBlockDetails(): Promise<Block> {
    return {
      id: "6c60ec01-8128-419e-988f-96a063ee2fea",
      name: "Github Trigger",
      description:
        "This block triggers on pull request events and outputs the event type and payload.",
      type: "Standard",
    };
  }

  async nextTutorialStep(): Promise<void> {
    console.log(`clicking next tutorial step`);
    await this.page.getByRole("button", { name: "Next" }).click();
  }

  async getBlocksToSkip(): Promise<string[]> {
    return [(await this.getGithubTriggerBlockDetails()).id];
  }

  async createDummyAgent() {
    await this.closeTutorial();
    await this.openBlocksPanel();
    const dictionaryBlock = await this.getDictionaryBlockDetails();

    const searchInput = this.page.locator(
      '[data-id="blocks-control-search-input"]',
    );

    const displayName = this.getDisplayName(dictionaryBlock.name);
    await searchInput.clear();

    await isVisible(this.page.getByText("Output"));

    await searchInput.fill(displayName);

    const blockCard = this.page.getByTestId(`block-name-${dictionaryBlock.id}`);
    if (await blockCard.isVisible()) {
      await blockCard.click();
      const blockInEditor = this.page.getByTestId(dictionaryBlock.id).first();
      expect(blockInEditor).toBeAttached();
    }

    await this.saveAgent("Test Agent", "Test Description");
    await expect(this.isRunButtonEnabled()).resolves.toBeTruthy();
  }
}
