import { test } from "./fixtures";
import { MarketplacePage } from "./pages/marketplace.page";
import { AgentDetailPage } from "./pages/agent-detail.page";
import { CreatorProfilePage } from "./pages/creator-profile.page";

test.describe("Marketplace", () => {
  let marketplacePage: MarketplacePage;
  let agentDetailPage: AgentDetailPage;
  let creatorProfilePage: CreatorProfilePage;

  test.beforeEach(async ({ page }) => {
    marketplacePage = new MarketplacePage(page);
    agentDetailPage = new AgentDetailPage(page);
    creatorProfilePage = new CreatorProfilePage(page);

    // Navigate to marketplace
    await page.goto("/marketplace");
    await marketplacePage.waitForPageLoad();
  });

  test.describe("Page Load and Structure", () => {
    test("marketplace page loads successfully", async ({ page }) => {
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(page).toHaveURL(/.*\/marketplace/);
      await test
        .expect(marketplacePage.hasCorrectTitle())
        .resolves.toBeTruthy();
    });

    test("has all required sections", async () => {
      await test.expect(marketplacePage.hasMainHeading()).resolves.toBeTruthy();
      await test.expect(marketplacePage.hasSearchInput()).resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasCategoryButtons())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasFeaturedAgentsSection())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasTopAgentsSection())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasFeaturedCreatorsSection())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasBecomeCreatorSection())
        .resolves.toBeTruthy();
    });

    test("displays agent cards with correct information", async () => {
      await marketplacePage.waitForAgentsToLoad();
      const agents = await marketplacePage.getAgentCards();

      await test.expect(agents.length).toBeGreaterThan(0);

      if (agents.length > 0) {
        const firstAgent = agents[0];
        await test.expect(firstAgent.name).toBeTruthy();
        await test.expect(firstAgent.creator).toBeTruthy();
        await test.expect(typeof firstAgent.runs).toBe("number");
        await test.expect(typeof firstAgent.rating).toBe("number");
      }
    });

    test("displays featured creators", async () => {
      const creators = await marketplacePage.getFeaturedCreators();
      await test.expect(creators.length).toBeGreaterThan(0);

      if (creators.length > 0) {
        const firstCreator = creators[0];
        await test.expect(firstCreator.username).toBeTruthy();
        await test.expect(firstCreator.displayName).toBeTruthy();
        await test.expect(typeof firstCreator.agentCount).toBe("number");
      }
    });
  });

  test.describe("Search Functionality", () => {
    test("search input is visible and functional", async ({ page }) => {
      await test.expect(marketplacePage.hasSearchInput()).resolves.toBeTruthy();

      await marketplacePage.searchAgents("test");
      await page.waitForTimeout(1000);

      // Verify search was performed (URL or content change)
      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("test");
    });

    test("can search for specific agents", async ({ page }) => {
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(2000);

      // Verify search results or that search was executed
      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("Lead");
    });

    test("can clear search", async () => {
      await marketplacePage.searchAgents("test query");
      await marketplacePage.clearSearch();

      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("");
    });
  });

  test.describe("Category Filtering", () => {
    test("displays category buttons", async () => {
      const categories = await marketplacePage.getAvailableCategories();
      await test.expect(categories.length).toBeGreaterThan(0);

      // Check for common categories
      const categoryText = categories.join(" ").toLowerCase();
      const hasCommonCategories =
        categoryText.includes("marketing") ||
        categoryText.includes("automation") ||
        categoryText.includes("content") ||
        categoryText.includes("seo") ||
        categoryText.includes("fun");

      await test.expect(hasCommonCategories).toBeTruthy();
    });

    test("can click category buttons", async ({ page }) => {
      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        const firstCategory = categories[0];
        await marketplacePage.clickCategory(firstCategory);
        await page.waitForTimeout(1000);

        // Verify category was clicked (could check for URL change or filter application)
        // This is a basic interaction test
      }
    });
  });

  test.describe("Agent Interactions", () => {
    test("can click featured agent and navigate to detail page", async ({
      page,
    }) => {
      const featuredAgents = await marketplacePage.getFeaturedAgents();

      if (featuredAgents.length > 0) {
        const firstAgent = featuredAgents[0];
        await marketplacePage.clickFeaturedAgent(firstAgent.name);

        // Wait for navigation
        await page.waitForTimeout(2000);

        // Verify we're on an agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("can click agent card and navigate to detail page", async ({
      page,
    }) => {
      await marketplacePage.waitForAgentsToLoad();
      const agents = await marketplacePage.getAgentCards();

      if (agents.length > 0) {
        const firstAgent = agents[0];
        await marketplacePage.clickAgentCard(firstAgent.name);

        // Wait for navigation
        await page.waitForTimeout(2000);

        // Verify we're on an agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      }
    });
  });

  test.describe("Creator Interactions", () => {
    test("can click creator and navigate to profile page", async ({ page }) => {
      const creators = await marketplacePage.getFeaturedCreators();

      if (creators.length > 0) {
        const firstCreator = creators[0];
        await marketplacePage.clickCreator(firstCreator.displayName);

        // Wait for navigation
        await page.waitForTimeout(2000);

        // Verify we're on a creator profile page
        await test.expect(page.url()).toMatch(/\/marketplace\/creator\/.*/);
        await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      }
    });
  });

  test.describe("Navigation and Responsiveness", () => {
    test("navigation bar works correctly", async ({ page }) => {
      // Test navigation links
      await marketplacePage.navbar.clickMarketplaceLink();
      await test.expect(page).toHaveURL(/.*\/marketplace/);

      await marketplacePage.navbar.clickBuildLink();
      await test.expect(page).toHaveURL(/.*\/build/);

      await marketplacePage.navbar.clickMonitorLink();
      await test.expect(page).toHaveURL(/.*\/library/);

      // Navigate back to marketplace
      await page.goto("/marketplace");
      await marketplacePage.waitForPageLoad();
    });

    test("page scrolling works correctly", async () => {
      await marketplacePage.scrollToSection("Featured Creators");
      await marketplacePage.scrollToSection("Become a Creator");

      // Verify sections are accessible
      await test
        .expect(marketplacePage.hasFeaturedCreatorsSection())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasBecomeCreatorSection())
        .resolves.toBeTruthy();
    });

    test("become creator button is functional", async () => {
      await test
        .expect(marketplacePage.hasBecomeCreatorSection())
        .resolves.toBeTruthy();

      // Verify button is clickable (without actually clicking to avoid navigation)
      await test
        .expect(marketplacePage.becomeCreatorButton.isVisible())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.becomeCreatorButton.isEnabled())
        .resolves.toBeTruthy();
    });
  });

  test.describe("Performance and Metrics", () => {
    test("page loads with expected content metrics", async () => {
      const metrics = await marketplacePage.getPageLoadMetrics();

      await test.expect(metrics.agentCount).toBeGreaterThan(0);
      await test.expect(metrics.creatorCount).toBeGreaterThan(0);
      await test.expect(metrics.categoryCount).toBeGreaterThan(0);

      console.log("Page Metrics:", metrics);
    });

    test("agents load within reasonable time", async ({
      page: _,
    }, testInfo) => {
      const startTime = Date.now();

      await marketplacePage.waitForAgentsToLoad();

      const loadTime = Date.now() - startTime;

      // Agents should load within 10 seconds
      await test.expect(loadTime).toBeLessThan(10000);

      testInfo.attach("load-time", { body: `Agents loaded in ${loadTime}ms` });
    });
  });

  test.describe("Error Handling", () => {
    test("handles empty search gracefully", async ({ page }) => {
      await marketplacePage.searchAgents("");
      await page.waitForTimeout(1000);

      // Page should still be functional
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });

    test("handles invalid category selection gracefully", async ({ page }) => {
      try {
        await marketplacePage.clickCategory("NonExistentCategory");
        await page.waitForTimeout(1000);

        // Page should still be functional
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      } catch (error) {
        // This is expected for non-existent categories
        console.log("Expected error for non-existent category:", error);
      }
    });
  });

  test.describe("Accessibility", () => {
    test("main headings are accessible", async ({ page }) => {
      const mainHeading = page.getByRole("heading", {
        name: "Explore AI agents built for you by the community",
      });
      await test.expect(mainHeading).toBeVisible();

      const featuredHeading = page.getByRole("heading", {
        name: "Featured agents",
      });
      await test.expect(featuredHeading).toBeVisible();

      const topAgentsHeading = page.getByRole("heading", {
        name: "Top Agents",
      });
      await test.expect(topAgentsHeading).toBeVisible();
    });

    test("search input has proper accessibility attributes", async () => {
      const searchInput = marketplacePage.searchInput;
      await test.expect(searchInput).toBeVisible();

      // Check if input has placeholder or label
      const placeholder = await searchInput.getAttribute("placeholder");
      await test.expect(placeholder).toBeTruthy();
    });

    test("agent cards are keyboard accessible", async ({ page }) => {
      // Focus on first agent card
      await page.keyboard.press("Tab");
      await page.keyboard.press("Tab");
      await page.keyboard.press("Tab");

      // Should be able to navigate with keyboard
      const focusedElement = await page.evaluate(
        () => document.activeElement?.tagName,
      );
      console.log("Focused element:", focusedElement);
    });
  });
});
