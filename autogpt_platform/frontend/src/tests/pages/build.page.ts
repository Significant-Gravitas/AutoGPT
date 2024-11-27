import { ElementHandle, Page } from "@playwright/test";
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

  // async hasBlock(block: Block): Promise<boolean> {
  //   // all blocks on graph have a ref id  getByTestId('rf__node-1') where -1 is the order it was added to the graph or the internal id after its been saved
  //   //so we can check by getting all elements with that testid and seeing if the textContent includes the block name
  //   const nodes = await this.page.locator('[data-testid^="rf__node-"]').all();
  //   console.log(`found ${nodes.length} nodes`);
  //   const nodeTexts = await Promise.all(
  //     nodes.map((node) => node.textContent()),
  //   );
  //   console.log(`nodeTexts: ${nodeTexts}`);
  //   const matches = nodeTexts.some((text) => text?.includes(block.name));
  //   console.log(`matches: ${matches}`);
  //   return matches;
  // }
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

  async fillBlockInputByPlaceholder(
    blockId: string,
    placeholder: string,
    value: string,
  ): Promise<void> {
    const block = await this.page.locator(`[data-blockid="${blockId}"]`);
    const input = await block.getByPlaceholder(placeholder);
    await input.fill(value);
  }

  async fillBlockInputByLabel(
    blockId: string,
    label: string,
    value: string,
  ): Promise<void> {
    throw new Error("Not implemented");
    // const block = await this.page.locator(`[data-blockid="${blockId}"]`);
    // const input = await block.getByLabel(label);
    // await input.fill(value);
  }

  async connectBlockOutputToBlockInput(
    blockOutputId: string,
    blockOutputName: string,
    blockInputId: string,
    blockInputName: string,
  ): Promise<void> {
    throw new Error("Not implemented");
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.page.waitForLoadState("networkidle", { timeout: 10_000 });
      return true;
    } catch (error) {
      return false;
    }
  }
}
