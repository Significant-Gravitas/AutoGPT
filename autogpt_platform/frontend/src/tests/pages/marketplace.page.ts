import { Page, Locator } from "@playwright/test";
import { BasePage } from "./base.page";

export interface Agent {
  id: string;
  name: string;
  description: string;
  creator: string;
  rating: number;
  runs: number;
  categories?: string[];
}

export interface Creator {
  username: string;
  displayName: string;
  description: string;
  agentCount: number;
  rating?: number;
  categories?: string[];
}

export class MarketplacePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Locators
  get searchInput(): Locator {
    return this.page.getByTestId("store-search-input");
  }

  get categoryButtons(): Locator {
    return this.page.locator('[data-testid*="category-"]');
  }

  get featuredAgentsSection(): Locator {
    return this.page.locator('h2:has-text("Featured agents")').locator("..");
  }

  get topAgentsSection(): Locator {
    return this.page.locator('h2:has-text("Top Agents")').locator("..");
  }

  get featuredCreatorsSection(): Locator {
    return this.page.locator('h2:has-text("Featured Creators")').locator("..");
  }

  get becomeCreatorButton(): Locator {
    return this.page.getByRole("button", { name: "Become a Creator" });
  }

  get agentCards(): Locator {
    return this.page.locator('button[data-testid*="agent-card"]');
  }

  get featuredAgentLinks(): Locator {
    return this.featuredAgentsSection.locator("a");
  }

  // Page load check
  async isLoaded(): Promise<boolean> {
    console.log("Checking if marketplace page is loaded");
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });

      // Check for main heading
      await this.page
        .getByRole("heading", {
          name: "Explore AI agents built for you by the community",
        })
        .waitFor({ state: "visible", timeout: 10_000 });

      // Check for search input
      await this.searchInput.waitFor({ state: "visible", timeout: 5_000 });

      return true;
    } catch (error) {
      console.error("Error checking if marketplace page is loaded:", error);
      return false;
    }
  }

  // Search functionality
  async searchAgents(query: string): Promise<void> {
    console.log(`Searching for agents with query: ${query}`);
    await this.searchInput.fill(query);
    await this.searchInput.press("Enter");
    // Wait for search results to load
    await this.page.waitForTimeout(1000);
  }

  async clearSearch(): Promise<void> {
    console.log("Clearing search input");
    await this.searchInput.clear();
  }

  // Category filtering
  async clickCategory(categoryName: string): Promise<void> {
    console.log(`Clicking category: ${categoryName}`);
    await this.page.locator(`text=${categoryName}`).first().click();
    await this.page.waitForTimeout(1000);
  }

  async getAvailableCategories(): Promise<string[]> {
    console.log("Getting available categories");
    const categories = await this.page
      .locator('div[role="button"]')
      .allTextContents();
    return categories.filter((cat) => cat.trim().length > 0);
  }

  // Agent interactions
  async getAgentCards(): Promise<Agent[]> {
    console.log("Getting agent cards from marketplace");
    const agents: Agent[] = [];

    // Get agent cards from both sections
    const topAgentCards = await this.topAgentsSection
      .locator('button[data-testid*="agent-card"]')
      .all();

    for (const card of topAgentCards) {
      try {
        const nameElement = await card.locator("h3").first();
        const name = await nameElement.textContent();

        const creatorElement = await card.locator('p:has-text("by ")').first();
        const creatorText = await creatorElement.textContent();
        const creator = creatorText?.replace("by ", "") || "";

        const descriptionElement = await card.locator("p").nth(1);
        const description = await descriptionElement.textContent();

        const runsElement = await card.locator('div:has-text("runs")');
        const runsText = await runsElement.textContent();
        const runs = parseInt(runsText?.match(/\d+/)?.[0] || "0");

        // Try to get rating
        const ratingElement = await card.locator('div:has-text(".")').first();
        const ratingText = await ratingElement.textContent();
        const rating = parseFloat(ratingText?.match(/\d+\.\d+/)?.[0] || "0");

        if (name) {
          agents.push({
            id: (await card.getAttribute("data-testid")) || "",
            name: name.trim(),
            description: description?.trim() || "",
            creator: creator.trim(),
            rating,
            runs,
          });
        }
      } catch (error) {
        console.error("Error parsing agent card:", error);
      }
    }

    return agents;
  }

  async getFeaturedAgents(): Promise<Agent[]> {
    console.log("Getting featured agents");
    const agents: Agent[] = [];

    const featuredLinks = await this.featuredAgentLinks.all();

    for (const link of featuredLinks) {
      try {
        const nameElement = await link.locator("h3").first();
        const name = await nameElement.textContent();

        const creatorElement = await link.locator('p:has-text("By ")').first();
        const creatorText = await creatorElement.textContent();
        const creator = creatorText?.replace("By ", "") || "";

        const descriptionElement = await link.locator("p").nth(1);
        const description = await descriptionElement.textContent();

        const runsElement = await link.locator('div:has-text("runs")');
        const runsText = await runsElement.textContent();
        const runs = parseInt(runsText?.match(/\d+/)?.[0] || "0");

        const ratingElement = await link.locator('p:has-text(".")').first();
        const ratingText = await ratingElement.textContent();
        const rating = parseFloat(ratingText?.match(/\d+\.\d+/)?.[0] || "0");

        if (name) {
          agents.push({
            id: (await link.getAttribute("href")) || "",
            name: name.trim(),
            description: description?.trim() || "",
            creator: creator.trim(),
            rating,
            runs,
          });
        }
      } catch (error) {
        console.error("Error parsing featured agent:", error);
      }
    }

    return agents;
  }

  async clickAgentCard(agentName: string): Promise<void> {
    console.log(`Clicking agent card: ${agentName}`);
    await this.page
      .getByRole("button", { name: new RegExp(agentName, "i") })
      .first()
      .click();
  }

  async clickFeaturedAgent(agentName: string): Promise<void> {
    console.log(`Clicking featured agent: ${agentName}`);
    await this.featuredAgentsSection
      .getByRole("link", { name: new RegExp(agentName, "i") })
      .first()
      .click();
  }

  // Creator interactions
  async getFeaturedCreators(): Promise<Creator[]> {
    console.log("Getting featured creators");
    const creators: Creator[] = [];

    const creatorElements = await this.featuredCreatorsSection
      .locator("div")
      .all();

    for (const element of creatorElements) {
      try {
        const nameElement = await element.locator("h3").first();
        const name = await nameElement.textContent();

        const descriptionElement = await element.locator("p").first();
        const description = await descriptionElement.textContent();

        const agentCountElement = await element.locator(
          'div:has-text("agents")',
        );
        const agentCountText = await agentCountElement.textContent();
        const agentCount = parseInt(agentCountText?.match(/\d+/)?.[0] || "0");

        if (name && description) {
          creators.push({
            username: name.trim(),
            displayName: name.trim(),
            description: description.trim(),
            agentCount,
          });
        }
      } catch (error) {
        console.error("Error parsing creator:", error);
      }
    }

    return creators;
  }

  async clickCreator(creatorName: string): Promise<void> {
    console.log(`Clicking creator: ${creatorName}`);
    await this.page.getByRole("heading", { name: creatorName }).click();
  }

  // Navigation checks
  async hasCorrectTitle(): Promise<boolean> {
    const title = await this.page.title();
    return title.includes("Marketplace") || title.includes("AutoGPT Platform");
  }

  async hasCorrectURL(): Promise<boolean> {
    const url = this.page.url();
    return url.includes("/marketplace");
  }

  // Content checks
  async hasMainHeading(): Promise<boolean> {
    try {
      await this.page
        .getByRole("heading", {
          name: "Explore AI agents built for you by the community",
        })
        .waitFor({ state: "visible", timeout: 5_000 });
      return true;
    } catch {
      return false;
    }
  }

  async hasSearchInput(): Promise<boolean> {
    return await this.searchInput.isVisible();
  }

  async hasCategoryButtons(): Promise<boolean> {
    const categories = await this.getAvailableCategories();
    return categories.length > 0;
  }

  async hasFeaturedAgentsSection(): Promise<boolean> {
    return await this.featuredAgentsSection.isVisible();
  }

  async hasTopAgentsSection(): Promise<boolean> {
    return await this.topAgentsSection.isVisible();
  }

  async hasFeaturedCreatorsSection(): Promise<boolean> {
    return await this.featuredCreatorsSection.isVisible();
  }

  async hasBecomeCreatorSection(): Promise<boolean> {
    return await this.becomeCreatorButton.isVisible();
  }

  // Utility methods
  async scrollToSection(sectionName: string): Promise<void> {
    console.log(`Scrolling to section: ${sectionName}`);
    await this.page
      .getByRole("heading", { name: sectionName })
      .scrollIntoViewIfNeeded();
  }

  async waitForAgentsToLoad(): Promise<void> {
    console.log("Waiting for agents to load");
    await this.page.waitForSelector('button[data-testid*="agent-card"]', {
      timeout: 10_000,
    });
  }

  async hasAgentCards(): Promise<boolean> {
    const agents = await this.getAgentCards();
    return agents.length > 0;
  }

  async getPageLoadMetrics(): Promise<{
    agentCount: number;
    creatorCount: number;
    categoryCount: number;
  }> {
    const agents = await this.getAgentCards();
    const creators = await this.getFeaturedCreators();
    const categories = await this.getAvailableCategories();

    return {
      agentCount: agents.length,
      creatorCount: creators.length,
      categoryCount: categories.length,
    };
  }
}
