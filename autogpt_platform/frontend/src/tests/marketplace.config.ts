// marketplace.config.ts
// NOTE: Marketplace tests use workaround for #8788 (double page reload)
// similar to build tests to ensure reliable loading in CI environments
export const MarketplaceTestConfig = {
  // Test timeouts
  timeouts: {
    // Increased timeouts for CI reliability (matching build test patterns)
    pageLoad: process.env.CI ? 30_000 : 10_000,
    navigation: process.env.CI ? 15_000 : 5_000,
    search: process.env.CI ? 10_000 : 3_000,
    agentLoad: process.env.CI ? 20_000 : 8_000,
    imageLoad: process.env.CI ? 20_000 : 10_000,
  },

  // Expected page elements
  expectedElements: {
    marketplace: {
      mainHeading: "Explore AI agents built for you by the community",
      sections: [
        "Featured agents",
        "Top Agents",
        "Featured Creators",
        "Become a Creator",
      ],
      categories: ["Marketing", "SEO", "Content Creation", "Automation", "Fun"],
    },
    agentDetail: {
      requiredElements: [
        "agent-name",
        "creator-info",
        "description",
        "download-button",
        "rating-info",
        "runs-info",
      ],
      optionalElements: [
        "version-info",
        "categories",
        "agent-images",
        "breadcrumb-navigation",
      ],
    },
    creatorProfile: {
      requiredElements: ["creator-name", "agents-section"],
      optionalElements: [
        "creator-handle",
        "creator-avatar",
        "description",
        "statistics",
        "top-categories",
      ],
    },
  },

  // Test data
  testData: {
    searchQueries: {
      valid: ["Lead", "test", "automation", "marketing", "content"],
      special: ["@test", "#hashtag", "test!@#", "test-agent"],
      edge: ["", "a".repeat(200), "zyxwvutsrqponmlkjihgfedcba"],
    },
    categories: {
      common: [
        "marketing",
        "automation",
        "content",
        "seo",
        "fun",
        "productivity",
      ],
      fallback: "Marketing", // Default category to test if others fail
    },
  },

  // URL patterns
  urlPatterns: {
    marketplace: /\/marketplace$/,
    agent: /\/marketplace\/agent\/.*\/.*/,
    creator: /\/marketplace\/creator\/.*/,
  },

  // Performance thresholds (adjusted for CI)
  performance: {
    maxPageLoadTime: process.env.CI ? 30_000 : 15_000,
    maxSearchTime: process.env.CI ? 10_000 : 5_000,
    maxFilterTime: process.env.CI ? 10_000 : 5_000,
    maxNavigationTime: process.env.CI ? 15_000 : 8_000,
  },

  // Viewport configurations for responsive testing
  viewports: {
    mobile: { width: 375, height: 667 },
    tablet: { width: 768, height: 1024 },
    desktop: { width: 1920, height: 1080 },
  },

  // Test selectors
  selectors: {
    marketplace: {
      searchInput: '[data-testid="store-search-input"]',
      agentCards: '[data-testid="store-card"]',
      creatorCards: '[data-testid="creator-card"]',
      heroSection: '[data-testid="hero-section"]',
      featuredAgents: 'h2:has-text("Featured agents") + *',
      topAgents: 'h2:has-text("Top Agents") + *',
      featuredCreators: 'h2:has-text("Featured Creators") + *',
      becomeCreator: 'button:has-text("Become a Creator")',
    },
    agentDetail: {
      agentName: "h1, h2, h3",
      creatorLink: 'a[href*="/marketplace/creator/"]',
      downloadButton: 'button:has-text("Download")',
      relatedAgents: '[data-testid="store-card"]',
      breadcrumb: 'nav, div:has-text("Marketplace")',
    },
    creatorProfile: {
      displayName: "h1",
      agentsSection: 'h2:has-text("Agents by") + *',
      agentCards: '[data-testid="store-card"]',
      breadcrumb: 'a:has-text("Store")',
    },
  },
};

// Helper functions for marketplace tests
export const MarketplaceTestHelpers = {
  // Wait for element with retry
  async waitForElementWithRetry(
    page: any,
    selector: string,
    maxRetries: number = 3,
    timeout: number = 5000,
  ): Promise<boolean> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        await page.waitForSelector(selector, { timeout });
        return true;
      } catch (_error) {
        if (i === maxRetries - 1) {
          console.error(
            `Failed to find element ${selector} after ${maxRetries} retries`,
          );
          return false;
        }
        await page.waitForTimeout(1000);
      }
    }
    return false;
  },

  // Extract agent data from card element
  async extractAgentFromCard(card: any): Promise<any> {
    try {
      const nameElement = await card.locator("h3").first();
      const name = (await nameElement.textContent())?.trim() || "";

      const creatorElement = await card.locator('p:has-text("by ")').first();
      const creatorText = await creatorElement.textContent();
      const creator = creatorText?.replace("by ", "").trim() || "";

      const descriptionElement = await card.locator("p").nth(1);
      const description =
        (await descriptionElement.textContent())?.trim() || "";

      const runsElement = await card.locator('div:has-text("runs")');
      const runsText = await runsElement.textContent();
      const runs = parseInt(runsText?.match(/\d+/)?.[0] || "0");

      const ratingElement = await card.locator('div:has-text(".")').first();
      const ratingText = await ratingElement.textContent();
      const rating = parseFloat(ratingText?.match(/\d+\.\d+/)?.[0] || "0");

      return {
        name,
        creator,
        description,
        runs,
        rating,
        isValid: name.length > 0 && creator.length > 0,
      };
    } catch (error) {
      console.error("Error extracting agent data:", error);
      return {
        name: "",
        creator: "",
        description: "",
        runs: 0,
        rating: 0,
        isValid: false,
      };
    }
  },

  // Validate URL structure
  validateUrl(
    url: string,
    expectedType: "marketplace" | "agent" | "creator",
  ): boolean {
    const patterns = MarketplaceTestConfig.urlPatterns;

    switch (expectedType) {
      case "marketplace":
        return patterns.marketplace.test(url);
      case "agent":
        return patterns.agent.test(url);
      case "creator":
        return patterns.creator.test(url);
      default:
        return false;
    }
  },

  // Generate test metrics
  generateTestMetrics(
    startTime: number,
    elementCounts: { [key: string]: number },
    errors: string[] = [],
  ) {
    const endTime = Date.now();
    const duration = endTime - startTime;

    return {
      duration,
      timestamp: new Date().toISOString(),
      elementCounts,
      errors,
      performance: {
        isWithinThreshold:
          duration < MarketplaceTestConfig.performance.maxPageLoadTime,
        loadTime: duration,
      },
    };
  },

  // Common assertions for marketplace tests
  async assertMarketplacePageStructure(page: any): Promise<void> {
    const config = MarketplaceTestConfig.expectedElements.marketplace;

    // Check main heading
    const mainHeading = page.getByRole("heading", { name: config.mainHeading });
    if (!(await mainHeading.isVisible())) {
      throw new Error("Main heading not visible");
    }

    // Check required sections
    for (const section of config.sections) {
      const sectionElement = page.getByRole("heading", { name: section });
      if (!(await sectionElement.isVisible())) {
        console.warn(`Section "${section}" not visible`);
      }
    }

    // Check search input
    const searchInput = page.getByTestId("store-search-input");
    if (!(await searchInput.isVisible())) {
      throw new Error("Search input not visible");
    }
  },

  // Wait for agents to load with retry
  async waitForAgentsLoad(page: any, minCount: number = 1): Promise<boolean> {
    const maxRetries = 5;
    const retryDelay = 2000;

    for (let i = 0; i < maxRetries; i++) {
      try {
        await page.waitForSelector('[data-testid="store-card"]', {
          timeout: 5000,
        });
        const agentCount = await page
          .locator('[data-testid="store-card"]')
          .count();

        if (agentCount >= minCount) {
          return true;
        }
      } catch (_error) {
        console.log(`Attempt ${i + 1}: Waiting for agents to load...`);
      }

      if (i < maxRetries - 1) {
        await page.waitForTimeout(retryDelay);
      }
    }

    console.warn("Agents did not load within expected time");
    return false;
  },

  // Clean up search state
  async cleanupSearchState(page: any): Promise<void> {
    try {
      const searchInput = page.getByTestId("store-search-input");
      await searchInput.clear();
      await page.waitForTimeout(500);
    } catch (error) {
      console.warn("Could not clear search input:", error);
    }
  },

  // Validate agent card data
  validateAgentData(agent: any): boolean {
    return (
      typeof agent.name === "string" &&
      agent.name.length > 0 &&
      typeof agent.creator === "string" &&
      agent.creator.length > 0 &&
      typeof agent.runs === "number" &&
      agent.runs >= 0 &&
      typeof agent.rating === "number" &&
      agent.rating >= 0 &&
      agent.rating <= 5
    );
  },

  // Common test data for reuse
  getTestSearchQueries(): string[] {
    return MarketplaceTestConfig.testData.searchQueries.valid;
  },

  getTestCategories(): string[] {
    return MarketplaceTestConfig.testData.categories.common;
  },

  // Performance measurement helper
  async measurePerformance<T>(
    operation: () => Promise<T>,
    operationName: string,
  ): Promise<{ result: T; duration: number; performanceLog: string }> {
    const startTime = Date.now();
    const result = await operation();
    const duration = Date.now() - startTime;

    const performanceLog = `${operationName} completed in ${duration}ms`;
    console.log(performanceLog);

    return { result, duration, performanceLog };
  },
};

// Export test tags for organization
export const MarketplaceTestTags = {
  SMOKE: "@smoke",
  REGRESSION: "@regression",
  PERFORMANCE: "@performance",
  ACCESSIBILITY: "@accessibility",
  SEARCH: "@search",
  FILTERING: "@filtering",
  NAVIGATION: "@navigation",
  RESPONSIVE: "@responsive",
};
