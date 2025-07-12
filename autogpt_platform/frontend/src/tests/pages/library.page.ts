import { Page } from "@playwright/test";
import { BasePage } from "./base.page";

export interface Agent {
  id: string;
  name: string;
  description: string;
  imageUrl?: string;
  seeRunsUrl: string;
  openInBuilderUrl: string;
}

export class LibraryPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async isLoaded(): Promise<boolean> {
    console.log(`checking if library page is loaded`);
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });

      // Wait for the search input to be visible
      await this.page.waitForSelector('[data-testid="library-textbox"]', {
        state: "visible",
        timeout: 10_000,
      });

      console.log("Library page is loaded successfully");
      return true;
    } catch (error) {
      console.log("Library page failed to load:", error);
      return false;
    }
  }

  async getAgentCount(): Promise<number> {
    console.log(`getting agent count`);
    try {
      const countText = await this.page
        .locator("text=/\\d+ agents/")
        .textContent();
      const match = countText?.match(/(\d+) agents/);
      return match ? parseInt(match[1], 10) : 0;
    } catch (error) {
      console.error("Error getting agent count:", error);
      return 0;
    }
  }

  async searchAgents(searchTerm: string): Promise<void> {
    console.log(`searching for agents with term: ${searchTerm}`);
    const searchInput = this.page.getByRole("textbox", {
      name: "Search agents",
    });
    await searchInput.fill(searchTerm);

    // Wait for search results to update
    await this.page.waitForTimeout(500);
  }

  async clearSearch(): Promise<void> {
    console.log(`clearing search`);
    try {
      // Look for the clear button (X icon)
      const clearButton = this.page.locator(".lucide.lucide-x");
      if (await clearButton.isVisible()) {
        await clearButton.click();
      } else {
        // If no clear button, clear the search input directly
        const searchInput = this.page.getByRole("textbox", {
          name: "Search agents",
        });
        await searchInput.fill("");
      }

      // Wait for results to update
      await this.page.waitForTimeout(500);
    } catch (error) {
      console.error("Error clearing search:", error);
    }
  }

  async selectSortOption(
    sortOption: "Creation Date" | "Last Modified",
  ): Promise<void> {
    console.log(`selecting sort option: ${sortOption}`);

    // Click the sort dropdown
    await this.page.getByRole("combobox").click();

    // Select the option
    await this.page.getByRole("option", { name: sortOption }).click();

    // Wait for results to update
    await this.page.waitForTimeout(500);
  }

  async getCurrentSortOption(): Promise<string> {
    console.log(`getting current sort option`);
    try {
      const sortCombobox = this.page.getByRole("combobox");
      const currentOption = await sortCombobox.textContent();
      return currentOption?.trim() || "";
    } catch (error) {
      console.error("Error getting current sort option:", error);
      return "";
    }
  }

  async openUploadDialog(): Promise<void> {
    console.log(`opening upload dialog`);
    await this.page.getByRole("button", { name: "Upload an agent" }).click();

    // Wait for dialog to appear
    await this.page.getByRole("dialog", { name: "Upload Agent" }).waitFor({
      state: "visible",
      timeout: 5_000,
    });
  }

  async closeUploadDialog(): Promise<void> {
    console.log(`closing upload dialog`);
    await this.page.getByRole("button", { name: "Close" }).click();

    // Wait for dialog to disappear
    await this.page.getByRole("dialog", { name: "Upload Agent" }).waitFor({
      state: "hidden",
      timeout: 5_000,
    });
  }

  async isUploadDialogVisible(): Promise<boolean> {
    console.log(`checking if upload dialog is visible`);
    try {
      const dialog = this.page.getByRole("dialog", { name: "Upload Agent" });
      return await dialog.isVisible();
    } catch {
      return false;
    }
  }

  async fillUploadForm(agentName: string, description: string): Promise<void> {
    console.log(
      `filling upload form with name: ${agentName}, description: ${description}`,
    );

    // Fill agent name
    await this.page
      .getByRole("textbox", { name: "Agent name" })
      .fill(agentName);

    // Fill description
    await this.page
      .getByRole("textbox", { name: "Description" })
      .fill(description);
  }

  async isUploadButtonEnabled(): Promise<boolean> {
    console.log(`checking if upload button is enabled`);
    try {
      const uploadButton = this.page.getByRole("button", {
        name: "Upload Agent",
      });
      return await uploadButton.isEnabled();
    } catch {
      return false;
    }
  }

  async getAgents(): Promise<Agent[]> {
    console.log(`getting all agents from library`);
    try {
      const agents: Agent[] = [];

      // Wait for agents to load
      await this.page.waitForTimeout(1000);

      // Get all agent cards
      const agentCards = await this.page.getByTestId("agent-card").all();

      console.log("Total number of agents : ", agentCards.length);

      for (const card of agentCards) {
        try {
          const nameElement = card.locator("h3");
          const descriptionElement = card.locator("p");
          const seeRunsLink = card.locator("a", { hasText: "See runs" });
          const openInBuilderLink = card.locator("a", {
            hasText: "Open in builder",
          });

          const name = await nameElement.textContent();
          const description = await descriptionElement.textContent();
          const seeRunsUrl = await seeRunsLink.getAttribute("href");
          const openInBuilderUrl = await openInBuilderLink.getAttribute("href");

          if (name && description && seeRunsUrl && openInBuilderUrl) {
            // Extract agent ID from the URL
            const idMatch = seeRunsUrl.match(/\/library\/agents\/([^\/]+)/);
            const id = idMatch ? idMatch[1] : "";

            agents.push({
              id,
              name: name.trim(),
              description: description.trim(),
              seeRunsUrl,
              openInBuilderUrl,
            });
          }
        } catch (error) {
          console.error("Error processing agent card:", error);
        }
      }

      console.log(`found ${agents.length} agents`);
      return agents;
    } catch (error) {
      console.error("Error getting agents:", error);
      return [];
    }
  }

  async clickAgent(agent: Agent): Promise<void> {
    console.log(`clicking on agent: ${agent.name}`);

    // Click on the agent name/title
    await this.page
      .getByRole("heading", { name: agent.name, level: 3 })
      .click();
  }

  async clickSeeRuns(agent: Agent): Promise<void> {
    console.log(`clicking see runs for agent: ${agent.name}`);

    // Find the "See runs" link for this specific agent
    const agentCard = this.page.locator(`[href="${agent.seeRunsUrl}"]`).first();
    await agentCard.click();
  }

  async clickOpenInBuilder(agent: Agent): Promise<void> {
    console.log(`clicking open in builder for agent: ${agent.name}`);

    // Find the "Open in builder" link for this specific agent
    const builderLink = this.page
      .locator(`[href="${agent.openInBuilderUrl}"]`)
      .first();
    await builderLink.click();
  }

  async isAgentVisible(agent: Agent): Promise<boolean> {
    console.log(`checking if agent ${agent.name} is visible`);
    try {
      const agentHeading = this.page.getByRole("heading", {
        name: agent.name,
        level: 3,
      });
      return await agentHeading.isVisible();
    } catch {
      return false;
    }
  }

  async waitForAgentsToLoad(): Promise<void> {
    console.log(`waiting for agents to load`);

    // Wait for either agents to appear or the "0 agents" text to appear
    await Promise.race([
      this.page
        .getByTestId("agent-card")
        .first()
        .waitFor({ state: "visible", timeout: 10_000 }),
      this.page
        .getByTestId("agents-count")
        .waitFor({ state: "visible", timeout: 10_000 }),
    ]);
  }

  async clickMonitoringLink(): Promise<void> {
    console.log(`clicking monitoring link in alert`);
    await this.page.getByRole("link", { name: "here" }).click();
  }

  async isMonitoringAlertVisible(): Promise<boolean> {
    console.log(`checking if monitoring alert is visible`);
    try {
      const alertText = this.page.locator("text=/Prefer the old experience/");
      return await alertText.isVisible();
    } catch {
      return false;
    }
  }

  async getSearchValue(): Promise<string> {
    console.log(`getting search input value`);
    try {
      const searchInput = this.page.getByRole("textbox", {
        name: "Search agents",
      });
      return await searchInput.inputValue();
    } catch {
      return "";
    }
  }

  async hasNoAgentsMessage(): Promise<boolean> {
    console.log(`checking for no agents message`);
    try {
      const noAgentsText = this.page.locator("text=/0 agents/");
      return await noAgentsText.isVisible();
    } catch {
      return false;
    }
  }

  async scrollToBottom(): Promise<void> {
    console.log(`scrolling to bottom to trigger pagination`);
    await this.page.keyboard.press("End");
    await this.page.waitForTimeout(1000);
  }

  async scrollDown(): Promise<void> {
    console.log(`scrolling down to trigger pagination`);
    await this.page.keyboard.press("PageDown");
    await this.page.waitForTimeout(1000);
  }

  async scrollToLoadMore(): Promise<void> {
    console.log(`scrolling to load more agents`);

    // Get initial agent count
    const initialCount = await this.getAgentCount();
    console.log(`Initial agent count: ${initialCount}`);

    // Scroll down to trigger pagination
    await this.scrollToBottom();

    // Wait for potential new agents to load
    await this.page.waitForTimeout(2000);

    // Check if more agents loaded
    const newCount = await this.getAgentCount();
    console.log(`New agent count after scroll: ${newCount}`);

    return;
  }

  async testPagination(): Promise<{
    initialCount: number;
    finalCount: number;
    hasMore: boolean;
  }> {
    console.log(`testing pagination functionality`);

    // Get initial count
    const initialCount = await this.getAgentCount();
    console.log(`Initial agent count: ${initialCount}`);

    // Scroll to load more
    await this.scrollToLoadMore();

    // Get final count
    const finalCount = await this.getAgentCount();
    console.log(`Final agent count: ${finalCount}`);

    const hasMore = finalCount > initialCount;

    return {
      initialCount,
      finalCount,
      hasMore,
    };
  }

  async getAgentsWithPagination(): Promise<Agent[]> {
    console.log(`getting all agents with pagination`);

    let allAgents: Agent[] = [];
    let previousCount = 0;
    let currentCount = 0;
    const maxAttempts = 5; // Prevent infinite loop
    let attempts = 0;

    do {
      previousCount = currentCount;

      // Get current agents
      const currentAgents = await this.getAgents();
      allAgents = currentAgents;
      currentCount = currentAgents.length;

      console.log(`Attempt ${attempts + 1}: Found ${currentCount} agents`);

      // Try to load more by scrolling
      await this.scrollToLoadMore();

      attempts++;
    } while (currentCount > previousCount && attempts < maxAttempts);

    console.log(`Total agents found with pagination: ${allAgents.length}`);
    return allAgents;
  }

  async waitForPaginationLoad(): Promise<void> {
    console.log(`waiting for pagination to load`);

    // Wait for any loading states to complete
    await this.page.waitForTimeout(1000);

    // Wait for agent count to stabilize
    let previousCount = 0;
    let currentCount = 0;
    let stableChecks = 0;
    const maxChecks = 10;

    while (stableChecks < 3 && stableChecks < maxChecks) {
      currentCount = await this.getAgentCount();

      if (currentCount === previousCount) {
        stableChecks++;
      } else {
        stableChecks = 0;
      }

      previousCount = currentCount;
      await this.page.waitForTimeout(500);
    }

    console.log(`Pagination load stabilized with ${currentCount} agents`);
  }

  async scrollAndWaitForNewAgents(): Promise<number> {
    console.log(`scrolling and waiting for new agents to load`);

    const initialCount = await this.getAgentCount();

    // Scroll down
    await this.scrollDown();

    // Wait for potential new agents
    await this.waitForPaginationLoad();

    const finalCount = await this.getAgentCount();
    const newAgentsLoaded = finalCount - initialCount;

    console.log(
      `Loaded ${newAgentsLoaded} new agents (${initialCount} -> ${finalCount})`,
    );

    return newAgentsLoaded;
  }

  async isPaginationWorking(): Promise<boolean> {
    console.log(`checking if pagination is working`);

    const newAgentsLoaded = await this.scrollAndWaitForNewAgents();

    return newAgentsLoaded > 0;
  }
}
