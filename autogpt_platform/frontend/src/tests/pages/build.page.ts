import { ElementHandle, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

interface Block {
  id: string;
  name: string;
  description: string;
}

export class BuildPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async closeTutorial(): Promise<void> {
    await this.page.getByRole("button", { name: "Skip Tutorial" }).click();
  }

  async openBlocksPanel(): Promise<void> {
    if (
      !(await this.page.getByTestId("blocks-control-blocks-label").isVisible())
    ) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
    }
  }

  async closeBlocksPanel(): Promise<void> {
    if (
      await this.page.getByTestId("blocks-control-blocks-label").isVisible()
    ) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
    }
  }

  async saveAgent(
    name: string = "Test Agent",
    description: string = "",
  ): Promise<void> {
    await this.page.getByTestId("blocks-control-save-button").click();
    await this.page.getByTestId("save-control-name-input").fill(name);
    await this.page
      .getByTestId("save-control-description-input")
      .fill(description);
    await this.page.getByTestId("save-control-save-agent-button").click();
  }

  async getBlocks(): Promise<Block[]> {
    try {
      const blocks = await this.page.locator('[data-id^="block-card-"]').all();

      console.log(`found ${blocks.length} blocks`);

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

            return {
              id,
              name: name.trim(),
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
      console.error("Error getting blocks:", error);
      return [];
    }
  }

  async addBlock(block: Block): Promise<void> {
    console.log(`adding block ${block.id} ${block.name} to agent`);
    await this.page.getByTestId(`block-name-${block.id}`).click();
  }

  async isRFNodeVisible(nodeId: string): Promise<boolean> {
    return await this.page.getByTestId(`rf__node-${nodeId}`).isVisible();
  }

  async hasBlock(block: Block): Promise<boolean> {
    try {
      // Use both ID and name for most precise matching
      const node = await this.page
        .locator(`[data-blockid="${block.id}"]`)
        .first();
      return await node.isVisible();
    } catch (error) {
      console.error("Error checking for block:", error);
      return false;
    }
  }

  async getBlockInputs(blockId: string): Promise<string[]> {
    try {
      const node = await this.page
        .locator(`[data-blockid="${blockId}"]`)
        .first();
      const inputsData = await node.getAttribute("data-inputs");
      return inputsData ? JSON.parse(inputsData) : [];
    } catch (error) {
      console.error("Error getting block inputs:", error);
      return [];
    }
  }

  async getBlockOutputs(blockId: string): Promise<string[]> {
    throw new Error("Not implemented");
    // try {
    //   const node = await this.page
    //     .locator(`[data-blockid="${blockId}"]`)
    //     .first();
    //   const outputsData = await node.getAttribute("data-outputs");
    //   return outputsData ? JSON.parse(outputsData) : [];
    // } catch (error) {
    //   console.error("Error getting block outputs:", error);
    //   return [];
    // }
  }

  async build_block_selector(
    blockId: string,
    dataId?: string,
  ): Promise<string> {
    let selector = dataId
      ? `[data-id="${dataId}"] [data-blockid="${blockId}"]`
      : `[data-blockid="${blockId}"]`;
    return selector;
  }

  async getBlockById(blockId: string, dataId?: string): Promise<Locator> {
    return await this.page.locator(
      await this.build_block_selector(blockId, dataId),
    );
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
    const block = await this.getBlockById(blockId, dataId);
    const input = await block.getByPlaceholder(placeholder);
    await input.fill(value);
  }

  async selectBlockInputValue(
    blockId: string,
    inputName: string,
    value: string,
    dataId?: string,
  ): Promise<void> {
    // First get the button that opens the dropdown
    const baseSelector = await this.build_block_selector(blockId, dataId);

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
    // throw new Error("Not implemented");
    const block = await this.getBlockById(blockId);
    const input = await block.getByLabel(label);
    await input.fill(value);
  }

  async connectBlockOutputToBlockInputViaDataId(
    blockOutputId: string,
    blockInputId: string,
  ): Promise<void> {
    try {
      // Locate the output element
      const outputElement = await this.page.locator(
        `[data-id="${blockOutputId}"]`,
      );
      // Locate the input element
      const inputElement = await this.page.locator(
        `[data-id="${blockInputId}"]`,
      );

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
    const startBlockBase = await this.build_block_selector(
      startBlockId,
      startDataId,
    );
    const endBlockBase = await this.build_block_selector(endBlockId, endDataId);
    // Use descendant combinator to find test-id at any depth
    const startBlockOutputSelector = `${startBlockBase} [data-testid="output-handle-${startBlockOutputName.toLowerCase()}"]`;
    const endBlockInputSelector = `${endBlockBase} [data-testid="input-handle-${endBlockInputName.toLowerCase()}"]`;

    // Log for debugging
    console.log("Start block selector:", startBlockOutputSelector);
    console.log("End block selector:", endBlockInputSelector);

    await this.page
      .locator(startBlockOutputSelector)
      .dragTo(this.page.locator(endBlockInputSelector));
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.page.waitForLoadState("networkidle", { timeout: 10_000 });
      return true;
    } catch (error) {
      return false;
    }
  }

  async isRunButtonEnabled(): Promise<boolean> {
    const runButton = this.page.locator('[data-id="primary-action-run-agent"]');
    return await runButton.isEnabled();
  }

  async runAgent(): Promise<void> {
    const runButton = this.page.locator('[data-id="primary-action-run-agent"]');
    await runButton.click();
  }

  async fillRunDialog(inputs: Record<string, string>): Promise<void> {
    for (const [key, value] of Object.entries(inputs)) {
      await this.page.getByTestId(`run-dialog-input-${key}`).fill(value);
    }
  }
  async clickRunDialogRunButton(): Promise<void> {
    await this.page.getByTestId("run-dialog-run-button").click();
  }

  async waitForCompletionBadge(): Promise<void> {
    await this.page.waitForSelector(
      '[data-id^="badge-"][data-id$="-COMPLETED"]',
    );
  }

  async waitForSaveButton(): Promise<void> {
    await this.page.waitForSelector(
      '[data-testid="blocks-control-save-button"]:not([disabled])',
    );
  }

  async isCompletionBadgeVisible(): Promise<boolean> {
    const completionBadge = this.page
      .locator('[data-id^="badge-"][data-id$="-COMPLETED"]')
      .first();
    return await completionBadge.isVisible();
  }
}
