import { Page, Locator } from "@playwright/test";
import { BasePage } from "./base.page";

export interface CreatorProfile {
  username: string;
  displayName: string;
  handle: string;
  description: string;
  agentCount: number;
  averageRating: number;
  totalRuns: number;
  topCategories: string[];
}

export interface CreatorAgent {
  name: string;
  description: string;
  rating: number;
  runs: number;
  imageUrl?: string;
}

export class CreatorProfilePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Locators
  get creatorDisplayName(): Locator {
    return this.page.locator("h1").first();
  }

  get creatorHandle(): Locator {
    return this.page.locator('div:has-text("@")').first();
  }

  get creatorAvatar(): Locator {
    return this.page.locator('img[alt*="avatar"], img[alt*="profile"]').first();
  }

  get creatorDescription(): Locator {
    return this.page
      .locator("p, div")
      .filter({ hasText: /About|Description/ })
      .locator("+ p, + div");
  }

  get aboutSection(): Locator {
    return this.page.locator('p:has-text("About")').locator("..");
  }

  get topCategoriesSection(): Locator {
    return this.page.locator('div:has-text("Top categories")').locator("..");
  }

  get averageRatingSection(): Locator {
    return this.page.locator('div:has-text("Average rating")').locator("..");
  }

  get totalRunsSection(): Locator {
    return this.page.locator('div:has-text("Number of runs")').locator("..");
  }

  get agentsSection(): Locator {
    return this.page.locator('h2:has-text("Agents by")').locator("..");
  }

  get agentCards(): Locator {
    return this.page.locator('button[data-testid*="agent-card"]');
  }

  get breadcrumbNavigation(): Locator {
    return this.page.locator("nav, div").filter({ hasText: /Store.*\/.*/ });
  }

  get categoryTags(): Locator {
    return this.topCategoriesSection.locator("li, span");
  }

  // Page load and validation
  async isLoaded(): Promise<boolean> {
    console.log("Checking if creator profile page is loaded");
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });

      // Check for creator display name
      await this.creatorDisplayName.waitFor({
        state: "visible",
        timeout: 10_000,
      });

      // Check for agents section
      await this.agentsSection.waitFor({ state: "visible", timeout: 5_000 });

      return true;
    } catch (error) {
      console.error("Error checking if creator profile page is loaded:", error);
      return false;
    }
  }

  async hasCorrectURL(creatorHandle: string): Promise<boolean> {
    const url = this.page.url();
    return url.includes(`/marketplace/creator/${creatorHandle}`);
  }

  async hasCorrectTitle(): Promise<boolean> {
    const title = await this.page.title();
    return (
      title.includes("AutoGPT Store") || title.includes("AutoGPT Marketplace")
    );
  }

  // Content extraction
  async getCreatorProfile(): Promise<CreatorProfile> {
    console.log("Extracting creator profile information");

    const displayName =
      (await this.creatorDisplayName.textContent())?.trim() || "";

    let handle = "";
    try {
      const handleText = await this.creatorHandle.textContent();
      handle = handleText?.replace("@", "").trim() || "";
    } catch (error) {
      console.log("Could not extract handle:", error);
    }

    let description = "";
    try {
      description = (await this.creatorDescription.textContent())?.trim() || "";
    } catch (error) {
      console.log("Could not extract description:", error);
    }

    // Extract average rating
    let averageRating = 0;
    try {
      const ratingText = await this.averageRatingSection.textContent();
      const ratingMatch = ratingText?.match(/(\d+\.?\d*)/);
      averageRating = ratingMatch ? parseFloat(ratingMatch[1]) : 0;
    } catch (error) {
      console.log("Could not extract average rating:", error);
    }

    // Extract total runs
    let totalRuns = 0;
    try {
      const runsText = await this.totalRunsSection.textContent();
      const runsMatch = runsText?.match(/(\d+)\s*runs?/);
      totalRuns = runsMatch ? parseInt(runsMatch[1]) : 0;
    } catch (error) {
      console.log("Could not extract total runs:", error);
    }

    // Extract top categories
    const topCategories: string[] = [];
    try {
      const categoryElements = await this.categoryTags.all();
      for (const element of categoryElements) {
        const categoryText = await element.textContent();
        if (categoryText && categoryText.trim()) {
          topCategories.push(categoryText.trim());
        }
      }
    } catch (error) {
      console.log("Could not extract categories:", error);
    }

    // Count agents
    const agentCount = await this.agentCards.count();

    return {
      username: handle || displayName.toLowerCase().replace(/\s+/g, "-"),
      displayName,
      handle,
      description,
      agentCount,
      averageRating,
      totalRuns,
      topCategories,
    };
  }

  async getCreatorAgents(): Promise<CreatorAgent[]> {
    console.log("Getting creator's agents");
    const agents: CreatorAgent[] = [];

    const agentCards = await this.agentCards.all();

    for (const card of agentCards) {
      try {
        const nameElement = await card.locator("h3").first();
        const name = (await nameElement.textContent())?.trim() || "";

        const descriptionElement = await card.locator("p").first();
        const description =
          (await descriptionElement.textContent())?.trim() || "";

        // Extract rating
        let rating = 0;
        try {
          const ratingElement = await card.locator('div:has-text(".")').first();
          const ratingText = await ratingElement.textContent();
          const ratingMatch = ratingText?.match(/(\d+\.?\d*)/);
          rating = ratingMatch ? parseFloat(ratingMatch[1]) : 0;
        } catch {
          // Rating extraction failed, use default
        }

        // Extract runs
        let runs = 0;
        try {
          const runsElement = await card.locator('div:has-text("runs")');
          const runsText = await runsElement.textContent();
          const runsMatch = runsText?.match(/(\d+)\s*runs/);
          runs = runsMatch ? parseInt(runsMatch[1]) : 0;
        } catch {
          // Runs extraction failed, use default
        }

        // Extract image URL
        let imageUrl = "";
        try {
          const imageElement = await card.locator("img").first();
          imageUrl = (await imageElement.getAttribute("src")) || "";
        } catch {
          // Image extraction failed, use default
        }

        if (name) {
          agents.push({
            name,
            description,
            rating,
            runs,
            imageUrl,
          });
        }
      } catch (error) {
        console.error("Error parsing agent card:", error);
      }
    }

    return agents;
  }

  // Interactions
  async clickAgent(agentName: string): Promise<void> {
    console.log(`Clicking agent: ${agentName}`);
    await this.page
      .getByRole("button", { name: new RegExp(agentName, "i") })
      .first()
      .click();
  }

  async navigateBackToStore(): Promise<void> {
    console.log("Navigating back to store");
    await this.page.getByRole("link", { name: "Store" }).click();
  }

  async scrollToAgentsSection(): Promise<void> {
    console.log("Scrolling to agents section");
    await this.agentsSection.scrollIntoViewIfNeeded();
  }

  // Content validation
  async hasCreatorDisplayName(): Promise<boolean> {
    const name = await this.creatorDisplayName.textContent();
    return name !== null && name.trim().length > 0;
  }

  async hasCreatorHandle(): Promise<boolean> {
    try {
      return await this.creatorHandle.isVisible();
    } catch {
      return false;
    }
  }

  async hasCreatorAvatar(): Promise<boolean> {
    try {
      return await this.creatorAvatar.isVisible();
    } catch {
      return false;
    }
  }

  async hasCreatorDescription(): Promise<boolean> {
    try {
      const description = await this.creatorDescription.textContent();
      return description !== null && description.trim().length > 0;
    } catch {
      return false;
    }
  }

  async hasTopCategoriesSection(): Promise<boolean> {
    try {
      return await this.topCategoriesSection.isVisible();
    } catch {
      return false;
    }
  }

  async hasAverageRatingSection(): Promise<boolean> {
    try {
      return await this.averageRatingSection.isVisible();
    } catch {
      return false;
    }
  }

  async hasTotalRunsSection(): Promise<boolean> {
    try {
      return await this.totalRunsSection.isVisible();
    } catch {
      return false;
    }
  }

  async hasAgentsSection(): Promise<boolean> {
    return await this.agentsSection.isVisible();
  }

  async hasAgents(): Promise<boolean> {
    const agentCount = await this.agentCards.count();
    return agentCount > 0;
  }

  async hasBreadcrumbNavigation(): Promise<boolean> {
    try {
      return await this.breadcrumbNavigation.isVisible();
    } catch {
      return false;
    }
  }

  // Utility methods
  async waitForAgentsLoad(): Promise<void> {
    console.log("Waiting for creator's agents to load");
    try {
      await this.page.waitForSelector('button[data-testid*="agent-card"]', {
        timeout: 10_000,
      });
    } catch {
      console.log("No agent cards found or timeout reached");
    }
  }

  async getPageMetrics(): Promise<{
    hasAllRequiredElements: boolean;
    agentCount: number;
    categoryCount: number;
    hasProfileInfo: boolean;
  }> {
    const agents = await this.getCreatorAgents();
    const profile = await this.getCreatorProfile();

    const hasAllRequiredElements =
      (await this.hasCreatorDisplayName()) && (await this.hasAgentsSection());

    const hasProfileInfo =
      (await this.hasCreatorDescription()) ||
      (await this.hasAverageRatingSection()) ||
      (await this.hasTotalRunsSection());

    return {
      hasAllRequiredElements,
      agentCount: agents.length,
      categoryCount: profile.topCategories.length,
      hasProfileInfo,
    };
  }

  async searchCreatorAgents(query: string): Promise<CreatorAgent[]> {
    console.log(`Searching creator's agents for: ${query}`);
    const allAgents = await this.getCreatorAgents();
    return allAgents.filter(
      (agent) =>
        agent.name.toLowerCase().includes(query.toLowerCase()) ||
        agent.description.toLowerCase().includes(query.toLowerCase()),
    );
  }

  async getAgentsByCategory(category: string): Promise<CreatorAgent[]> {
    console.log(`Getting creator's agents by category: ${category}`);
    // This would require additional DOM structure to filter by category
    // For now, return all agents
    return await this.getCreatorAgents();
  }
}
