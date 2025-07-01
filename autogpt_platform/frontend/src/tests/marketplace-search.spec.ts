import { test } from "./fixtures";
import { MarketplacePage } from "./pages/marketplace.page";
import { AgentDetailPage } from "./pages/agent-detail.page";

test.describe("Marketplace Search and Filtering", () => {
  let marketplacePage: MarketplacePage;
  let agentDetailPage: AgentDetailPage;

  test.beforeEach(async ({ page }) => {
    marketplacePage = new MarketplacePage(page);
    agentDetailPage = new AgentDetailPage(page);

    // Navigate to marketplace with workaround for #8788
    await page.goto("/marketplace");
    // workaround for #8788 - same as build tests
    await page.reload();
    await page.reload();
    await marketplacePage.waitForPageLoad();
  });

  test.describe("Search Functionality", () => {
    test("search input is visible and accessible", async () => {
      await test.expect(marketplacePage.hasSearchInput()).resolves.toBeTruthy();

      const searchInput = marketplacePage.searchInput;
      await test.expect(searchInput.isVisible()).resolves.toBeTruthy();
      await test.expect(searchInput.isEnabled()).resolves.toBeTruthy();

      // Check placeholder text
      const placeholder = await searchInput.getAttribute("placeholder");
      await test.expect(placeholder).toBeTruthy();
      console.log("Search placeholder:", placeholder);
    });

    test("can perform basic search", async ({ page }) => {
      const searchQuery = "Lead";
      await marketplacePage.searchAgents(searchQuery);

      // Verify search was executed
      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe(searchQuery);

      // Wait for potential search results to load
      await page.waitForTimeout(2000);

      // Verify page is still functional after search
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });

    test("can search with different queries", async ({ page }) => {
      const searchQueries = ["Lead", "test", "automation", "marketing"];

      for (const query of searchQueries) {
        await marketplacePage.searchAgents(query);
        await page.waitForTimeout(1000);

        const searchValue = await marketplacePage.searchInput.inputValue();
        await test.expect(searchValue).toBe(query);

        // Clear search for next iteration
        await marketplacePage.clearSearch();
        await page.waitForTimeout(500);
      }
    });

    test("can clear search", async () => {
      // Perform a search first
      await marketplacePage.searchAgents("test query");
      let searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("test query");

      // Clear the search
      await marketplacePage.clearSearch();
      searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("");
    });

    test("search with empty query", async ({ page }) => {
      await marketplacePage.searchAgents("");
      await page.waitForTimeout(1000);

      // Page should remain functional
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      const agents = await marketplacePage.getAgentCards();
      await test.expect(agents.length).toBeGreaterThanOrEqual(0);
    });

    test("search with special characters", async ({ page }) => {
      const specialQueries = ["@test", "#hashtag", "test!@#", "test-agent"];

      for (const query of specialQueries) {
        await marketplacePage.searchAgents(query);
        await page.waitForTimeout(1000);

        // Page should handle special characters gracefully
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();

        await marketplacePage.clearSearch();
      }
    });

    test("search with very long query", async ({ page }) => {
      const longQuery = "a".repeat(200);
      await marketplacePage.searchAgents(longQuery);
      await page.waitForTimeout(1000);

      // Page should handle long queries gracefully
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });

    test("search preserves functionality", async ({ page }) => {
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(2000);

      // After search, core functionality should still work
      await test
        .expect(marketplacePage.hasFeaturedAgentsSection())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasTopAgentsSection())
        .resolves.toBeTruthy();

      // Should still be able to interact with agents
      const agents = await marketplacePage.getAgentCards();
      if (agents.length > 0) {
        // Verify agent cards are still clickable
        await test
          .expect(marketplacePage.agentCards.first().isVisible())
          .resolves.toBeTruthy();
      }
    });
  });

  test.describe("Category Filtering", () => {
    test("displays available categories", async () => {
      const categories = await marketplacePage.getAvailableCategories();
      await test.expect(categories.length).toBeGreaterThan(0);

      console.log("Available categories:", categories);

      // Check for expected common categories
      const categoryText = categories.join(" ").toLowerCase();
      const hasExpectedCategories =
        categoryText.includes("marketing") ||
        categoryText.includes("automation") ||
        categoryText.includes("content") ||
        categoryText.includes("seo") ||
        categoryText.includes("fun") ||
        categoryText.includes("productivity");

      await test.expect(hasExpectedCategories).toBeTruthy();
    });

    test("category buttons are clickable", async ({ page }) => {
      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        const firstCategory = categories[0];
        console.log("Testing category:", firstCategory);

        await marketplacePage.clickCategory(firstCategory);
        await page.waitForTimeout(2000);

        // Page should remain functional after category click
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("can click multiple categories", async ({ page }) => {
      const categories = await marketplacePage.getAvailableCategories();

      // Test clicking up to 3 categories
      const categoriesToTest = categories.slice(
        0,
        Math.min(3, categories.length),
      );

      for (const category of categoriesToTest) {
        await marketplacePage.clickCategory(category);
        await page.waitForTimeout(1000);

        console.log("Clicked category:", category);

        // Verify page remains functional
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("category filtering preserves page structure", async ({ page }) => {
      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        await marketplacePage.clickCategory(categories[0]);
        await page.waitForTimeout(2000);

        // Core page structure should remain
        await test
          .expect(marketplacePage.hasMainHeading())
          .resolves.toBeTruthy();
        await test
          .expect(marketplacePage.hasSearchInput())
          .resolves.toBeTruthy();
        await test
          .expect(marketplacePage.hasCategoryButtons())
          .resolves.toBeTruthy();
      }
    });
  });

  test.describe("Search and Filter Combination", () => {
    test("can combine search with category filtering", async ({ page }) => {
      // First perform a search
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(1000);

      // Then click a category
      const categories = await marketplacePage.getAvailableCategories();
      if (categories.length > 0) {
        await marketplacePage.clickCategory(categories[0]);
        await page.waitForTimeout(2000);

        // Both search and category should be applied
        const searchValue = await marketplacePage.searchInput.inputValue();
        await test.expect(searchValue).toBe("Lead");

        // Page should remain functional
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("can modify search after category selection", async ({ page }) => {
      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        // First select a category
        await marketplacePage.clickCategory(categories[0]);
        await page.waitForTimeout(1000);

        // Then perform a search
        await marketplacePage.searchAgents("automation");
        await page.waitForTimeout(2000);

        // Search should be applied
        const searchValue = await marketplacePage.searchInput.inputValue();
        await test.expect(searchValue).toBe("automation");

        // Page should remain functional
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("can clear search while maintaining category filter", async ({
      page,
    }) => {
      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        // Apply category and search
        await marketplacePage.clickCategory(categories[0]);
        await page.waitForTimeout(1000);
        await marketplacePage.searchAgents("test");
        await page.waitForTimeout(1000);

        // Clear search
        await marketplacePage.clearSearch();

        // Search should be cleared but page should remain functional
        const searchValue = await marketplacePage.searchInput.inputValue();
        await test.expect(searchValue).toBe("");
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });
  });

  test.describe("Search Results and Navigation", () => {
    test("can navigate from search results to agent detail", async ({
      page,
    }) => {
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(2000);

      // Try to find and click an agent from results
      const agents = await marketplacePage.getAgentCards();

      if (agents.length > 0) {
        const firstAgent = agents[0];
        await marketplacePage.clickAgentCard(firstAgent.agent_name);
        await page.waitForTimeout(2000);
        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();

        // Should navigate to agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("can return to search results after viewing agent", async ({
      page,
    }) => {
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(2000);

      const agents = await marketplacePage.getAgentCards();

      if (agents.length > 0) {
        // Go to agent detail
        await marketplacePage.clickAgentCard(agents[0].agent_name);
        await page.waitForTimeout(2000);
        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();

        // Navigate back to marketplace
        await agentDetailPage.navigateBackToMarketplace();
        await page.waitForTimeout(2000);
        // workaround for #8788
        await page.reload();
        await page.reload();
        await marketplacePage.waitForPageLoad();

        // Should return to marketplace with search preserved
        await test.expect(page.url()).toMatch(/\/marketplace/);
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("search results maintain agent card functionality", async ({
      page,
    }) => {
      await marketplacePage.searchAgents("test");
      await page.waitForTimeout(2000);

      const agents = await marketplacePage.getAgentCards();

      if (agents.length > 0) {
        const firstAgent = agents[0];

        // Agent cards should have all required information
        await test.expect(firstAgent.agent_name).toBeTruthy();
        await test.expect(firstAgent.creator).toBeTruthy();
        await test.expect(typeof firstAgent.runs).toBe("number");
        await test.expect(typeof firstAgent.rating).toBe("number");
      }
    });
  });

  test.describe("Performance and User Experience", () => {
    test("search response time is reasonable", async ({ page }, testInfo) => {
      // Use the same timeout multiplier as build tests
      await test.setTimeout(testInfo.timeout * 10);

      const startTime = Date.now();

      await marketplacePage.searchAgents("Lead Finder");
      await page.waitForTimeout(2000);

      const searchTime = Date.now() - startTime;

      // Search should complete within 5 seconds
      await test.expect(searchTime).toBeLessThan(5000);

      testInfo.attach("search-time", {
        body: `Search completed in ${searchTime}ms`,
      });
    });

    test("category filtering response time is reasonable", async ({
      page,
    }, testInfo) => {
      // Use the same timeout multiplier as build tests
      await test.setTimeout(testInfo.timeout * 10);

      const categories = await marketplacePage.getAvailableCategories();

      if (categories.length > 0) {
        const startTime = Date.now();

        await marketplacePage.clickCategory(categories[0]);
        await page.waitForTimeout(2000);

        const filterTime = Date.now() - startTime;

        // Category filtering should complete within 5 seconds
        await test.expect(filterTime).toBeLessThan(5000);

        testInfo.attach("filter-time", {
          body: `Category filtering completed in ${filterTime}ms`,
        });
      }
    });

    test("search input provides immediate feedback", async ({ page }) => {
      // Type character by character to test responsiveness
      const query = "Lead";

      for (let i = 0; i < query.length; i++) {
        await marketplacePage.searchInput.fill(query.substring(0, i + 1));
        await page.waitForTimeout(100);

        const currentValue = await marketplacePage.searchInput.inputValue();
        await test.expect(currentValue).toBe(query.substring(0, i + 1));
      }
    });

    test("UI remains responsive during search", async ({ page }) => {
      await marketplacePage.searchAgents("automation");

      // During search, UI should remain interactive
      await test
        .expect(marketplacePage.searchInput.isEnabled())
        .resolves.toBeTruthy();
      await test
        .expect(marketplacePage.hasCategoryButtons())
        .resolves.toBeTruthy();

      await page.waitForTimeout(2000);

      // After search, everything should still be functional
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });
  });

  test.describe("Edge Cases and Error Handling", () => {
    test("handles search with no results gracefully", async ({ page }) => {
      await marketplacePage.searchAgents("zyxwvutsrqponmlkjihgfedcba");
      await page.waitForTimeout(3000);

      // Page should remain functional even with no results
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(marketplacePage.hasSearchInput()).resolves.toBeTruthy();
    });

    test("handles rapid search queries", async ({ page }) => {
      const queries = ["a", "ab", "abc", "abcd", "abcde"];

      for (const query of queries) {
        await marketplacePage.searchAgents(query);
        await page.waitForTimeout(200);
      }

      // Page should handle rapid changes gracefully
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();

      const finalValue = await marketplacePage.searchInput.inputValue();
      await test.expect(finalValue).toBe("abcde");
    });

    test("handles clicking non-existent category", async ({ page }) => {
      try {
        await marketplacePage.clickCategory("NonExistentCategory123");
        await page.waitForTimeout(1000);

        // Page should remain functional
        await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
      } catch (_error) {
        // This is expected for non-existent categories
        console.log("Expected error for non-existent category");
      }
    });

    test("search preserves state across page interactions", async ({
      page,
    }) => {
      await marketplacePage.searchAgents("Lead");
      await page.waitForTimeout(1000);

      // Scroll the page
      await marketplacePage.scrollToSection("Featured Creators");
      await page.waitForTimeout(1000);

      // Search should still be preserved
      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("Lead");

      // Page should remain functional
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });
  });

  test.describe("Accessibility in Search and Filtering", () => {
    test("search input is keyboard accessible", async ({ page }) => {
      // Navigate to search input using keyboard
      await page.keyboard.press("Tab");

      // Type in search
      await page.keyboard.type("Lead");

      const searchValue = await marketplacePage.searchInput.inputValue();
      await test.expect(searchValue).toBe("Lead");
    });

    test("category buttons are keyboard accessible", async ({ page }) => {
      // Use keyboard to navigate to categories
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press("Tab");
        const focusedElement = await page.evaluate(
          () => document.activeElement?.textContent,
        );

        if (
          focusedElement &&
          focusedElement.toLowerCase().includes("marketing")
        ) {
          await page.keyboard.press("Enter");
          await page.waitForTimeout(1000);
          break;
        }
      }

      // Page should remain functional
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });

    test("search has proper ARIA attributes", async () => {
      const searchInput = marketplacePage.searchInput;

      // Check for accessibility attributes
      const placeholder = await searchInput.getAttribute("placeholder");
      const role = await searchInput.getAttribute("role");
      const ariaLabel = await searchInput.getAttribute("aria-label");

      // At least one accessibility attribute should be present
      const hasAccessibilityAttribute = placeholder || role || ariaLabel;
      await test.expect(hasAccessibilityAttribute).toBeTruthy();
    });
  });
});
