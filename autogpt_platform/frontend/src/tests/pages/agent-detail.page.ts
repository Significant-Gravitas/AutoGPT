import { Page, Locator } from "@playwright/test";
import { BasePage } from "./base.page";

export interface AgentDetails {
  name: string;
  creator: string;
  description: string;
  rating: number;
  runs: number;
  categories: string[];
  version: string;
  lastUpdated: string;
}

export interface RelatedAgent {
  name: string;
  creator: string;
  description: string;
  rating: number;
  runs: number;
}

export class AgentDetailPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Locators
  get agentName(): Locator {
    return this.page.locator("h1, h2, h3").first();
  }

  get creatorLink(): Locator {
    return this.page.locator('a[href*="/marketplace/creator/"]');
  }

  get agentDescription(): Locator {
    return this.page
      .locator('div:has-text("Description")')
      .locator("+ div, + p");
  }

  get downloadButton(): Locator {
    return this.page.getByRole("button", { name: "Download agent" });
  }

  get ratingSection(): Locator {
    return this.page.locator('div:has(img[alt*="Icon"])').first();
  }

  get runsCount(): Locator {
    return this.page.locator('div:has-text("runs")');
  }

  get categoriesSection(): Locator {
    return this.page.locator('div:has-text("Categories")').locator("+ div");
  }

  get versionInfo(): Locator {
    return this.page.locator('div:has-text("Version")');
  }

  get lastUpdatedInfo(): Locator {
    return this.page.locator('div:has-text("Last updated")');
  }

  get breadcrumbNavigation(): Locator {
    return this.page
      .locator("nav, div")
      .filter({ hasText: /Marketplace.*\/.*\/.*/ });
  }

  get agentImages(): Locator {
    return this.page.locator('img[alt*="Image"]');
  }

  get otherAgentsByCreatorSection(): Locator {
    return this.page.locator('h2:has-text("Other agents by")').locator("..");
  }

  get similarAgentsSection(): Locator {
    return this.page.locator('h2:has-text("Similar agents")').locator("..");
  }

  get relatedAgentCards(): Locator {
    return this.page.locator('button[data-testid*="agent-card"]');
  }

  // Page load and validation - simplified like build page
  async isLoaded(): Promise<boolean> {
    console.log("Checking if agent detail page is loaded");
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  async hasCorrectURL(creator: string, agentName: string): Promise<boolean> {
    const url = this.page.url();
    const expectedPattern = `/marketplace/agent/${creator}/${agentName.toLowerCase().replace(/\s+/g, "-")}`;
    return (
      url.includes(expectedPattern) ||
      url.includes(`/marketplace/agent/${creator}/`)
    );
  }

  async hasCorrectTitle(): Promise<boolean> {
    const title = await this.page.title();
    return (
      title.includes("AutoGPT") &&
      (title.includes("Marketplace") || title.includes("Store"))
    );
  }

  // Content extraction
  async getAgentDetails(): Promise<AgentDetails> {
    console.log("Extracting agent details");

    const name = (await this.agentName.textContent())?.trim() || "";

    const creatorText = await this.creatorLink.textContent();
    const creator = creatorText?.trim() || "";

    const description =
      (await this.agentDescription.textContent())?.trim() || "";

    // Extract rating
    let rating = 0;
    try {
      const ratingText = await this.ratingSection.textContent();
      const ratingMatch = ratingText?.match(/(\d+\.?\d*)/);
      rating = ratingMatch ? parseFloat(ratingMatch[1]) : 0;
    } catch (error) {
      console.log("Could not extract rating:", error);
    }

    // Extract runs count
    let runs = 0;
    try {
      const runsText = await this.runsCount.textContent();
      const runsMatch = runsText?.match(/(\d+)\s*runs/);
      runs = runsMatch ? parseInt(runsMatch[1]) : 0;
    } catch (error) {
      console.log("Could not extract runs count:", error);
    }

    // Extract categories
    let categories: string[] = [];
    try {
      const categoriesText = await this.categoriesSection.textContent();
      categories = categoriesText
        ? categoriesText.split(/[,\s]+/).filter((c) => c.trim())
        : [];
    } catch (error) {
      console.log("Could not extract categories:", error);
    }

    // Extract version
    let version = "";
    try {
      const versionText = await this.versionInfo.textContent();
      const versionMatch = versionText?.match(/Version\s+(\d+)/);
      version = versionMatch ? versionMatch[1] : "";
    } catch (error) {
      console.log("Could not extract version:", error);
    }

    // Extract last updated
    let lastUpdated = "";
    try {
      const lastUpdatedText = await this.lastUpdatedInfo.textContent();
      lastUpdated = lastUpdatedText?.replace("Last updated", "").trim() || "";
    } catch (error) {
      console.log("Could not extract last updated:", error);
    }

    return {
      name,
      creator,
      description,
      rating,
      runs,
      categories,
      version,
      lastUpdated,
    };
  }

  async getRelatedAgents(): Promise<RelatedAgent[]> {
    console.log("Getting related agents");
    const relatedAgents: RelatedAgent[] = [];

    const agentCards = await this.relatedAgentCards.all();

    for (const card of agentCards) {
      try {
        const nameElement = await card.locator("h3").first();
        const name = (await nameElement.textContent())?.trim() || "";

        const creatorElement = await card.locator('p:has-text("by ")').first();
        const creatorText = await creatorElement.textContent();
        const creator = creatorText?.replace("by ", "").trim() || "";

        const descriptionElement = await card.locator("p").nth(1);
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

        if (name) {
          relatedAgents.push({
            name,
            creator,
            description,
            rating,
            runs,
          });
        }
      } catch (error) {
        console.error("Error parsing related agent card:", error);
      }
    }

    return relatedAgents;
  }

  // Interactions
  async clickDownloadAgent(): Promise<void> {
    console.log("Clicking download agent button");
    await this.downloadButton.click();
  }

  async clickCreatorLink(): Promise<void> {
    console.log("Clicking creator link");
    await this.creatorLink.click();
  }

  async clickRelatedAgent(agentName: string): Promise<void> {
    console.log(`Clicking related agent: ${agentName}`);
    await this.page
      .getByRole("button", { name: new RegExp(agentName, "i") })
      .first()
      .click();
  }

  async navigateBackToMarketplace(): Promise<void> {
    console.log("Navigating back to marketplace");
    await this.page.getByRole("link", { name: "Marketplace" }).click();
  }

  // Content validation
  async hasAgentName(): Promise<boolean> {
    const name = await this.agentName.textContent();
    return name !== null && name.trim().length > 0;
  }

  async hasCreatorInfo(): Promise<boolean> {
    return await this.creatorLink.isVisible();
  }

  async hasDescription(): Promise<boolean> {
    const description = await this.agentDescription.textContent();
    return description !== null && description.trim().length > 0;
  }

  async hasDownloadButton(): Promise<boolean> {
    return await this.downloadButton.isVisible();
  }

  async hasRatingInfo(): Promise<boolean> {
    return await this.ratingSection.isVisible();
  }

  async hasRunsInfo(): Promise<boolean> {
    return await this.runsCount.isVisible();
  }

  async hasCategoriesInfo(): Promise<boolean> {
    try {
      return await this.categoriesSection.isVisible();
    } catch {
      return false;
    }
  }

  async hasVersionInfo(): Promise<boolean> {
    try {
      return await this.versionInfo.isVisible();
    } catch {
      return false;
    }
  }

  async hasAgentImages(): Promise<boolean> {
    const images = await this.agentImages.count();
    return images > 0;
  }

  async hasBreadcrumbNavigation(): Promise<boolean> {
    try {
      return await this.breadcrumbNavigation.isVisible();
    } catch {
      return false;
    }
  }

  async hasOtherAgentsByCreatorSection(): Promise<boolean> {
    try {
      return await this.otherAgentsByCreatorSection.isVisible();
    } catch {
      return false;
    }
  }

  async hasSimilarAgentsSection(): Promise<boolean> {
    try {
      return await this.similarAgentsSection.isVisible();
    } catch {
      return false;
    }
  }

  // Utility methods
  async scrollToSection(sectionName: string): Promise<void> {
    console.log(`Scrolling to section: ${sectionName}`);
    await this.page
      .getByRole("heading", { name: new RegExp(sectionName, "i") })
      .scrollIntoViewIfNeeded();
  }

  async waitForImagesLoad(): Promise<void> {
    console.log("Waiting for images to load");
    await this.page.waitForLoadState("networkidle", { timeout: 10_000 });
  }

  async getPageMetrics(): Promise<{
    hasAllRequiredElements: boolean;
    relatedAgentsCount: number;
    imageCount: number;
  }> {
    const relatedAgents = await this.getRelatedAgents();
    const imageCount = await this.agentImages.count();

    const hasAllRequiredElements =
      (await this.hasAgentName()) &&
      (await this.hasCreatorInfo()) &&
      (await this.hasDescription()) &&
      (await this.hasDownloadButton());

    return {
      hasAllRequiredElements,
      relatedAgentsCount: relatedAgents.length,
      imageCount,
    };
  }
}
