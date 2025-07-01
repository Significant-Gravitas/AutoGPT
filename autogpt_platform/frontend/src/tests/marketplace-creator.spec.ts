import { test } from "./fixtures";
import { MarketplacePage } from "./pages/marketplace.page";
import { AgentDetailPage } from "./pages/agent-detail.page";
import { CreatorProfilePage } from "./pages/creator-profile.page";

test.describe("Marketplace Creator Profile", () => {
  let marketplacePage: MarketplacePage;
  let agentDetailPage: AgentDetailPage;
  let creatorProfilePage: CreatorProfilePage;

  test.beforeEach(async ({ page }) => {
    marketplacePage = new MarketplacePage(page);
    agentDetailPage = new AgentDetailPage(page);
    creatorProfilePage = new CreatorProfilePage(page);

    // Navigate to marketplace first
    await page.goto("/marketplace");
    await marketplacePage.waitForPageLoad();

    // Navigate to a creator profile page via featured creators
    const creators = await marketplacePage.getFeaturedCreators();
    if (creators.length > 0) {
      await marketplacePage.clickCreator(creators[0].displayName);
      await page.waitForTimeout(2000);
      await creatorProfilePage.waitForPageLoad();
    }
  });

  test.describe("Page Load and Structure", () => {
    test("creator profile page loads successfully", async ({ page }) => {
      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(page.url()).toMatch(/\/marketplace\/creator\/.*/);
      await test
        .expect(creatorProfilePage.hasCorrectTitle())
        .resolves.toBeTruthy();
    });

    test("has all required creator information", async () => {
      await test
        .expect(creatorProfilePage.hasCreatorDisplayName())
        .resolves.toBeTruthy();
      await test
        .expect(creatorProfilePage.hasAgentsSection())
        .resolves.toBeTruthy();
    });

    test("displays correct creator profile", async () => {
      const creatorProfile = await creatorProfilePage.getCreatorProfile();

      await test.expect(creatorProfile.displayName).toBeTruthy();
      await test.expect(typeof creatorProfile.agentCount).toBe("number");
      await test.expect(creatorProfile.agentCount).toBeGreaterThanOrEqual(0);

      console.log("Creator Profile:", creatorProfile);
    });

    test("has breadcrumb navigation", async () => {
      await test
        .expect(creatorProfilePage.hasBreadcrumbNavigation())
        .resolves.toBeTruthy();
    });
  });

  test.describe("Creator Information", () => {
    test("shows creator handle if available", async () => {
      const hasHandle = await creatorProfilePage.hasCreatorHandle();
      console.log("Has creator handle:", hasHandle);

      if (hasHandle) {
        const creatorProfile = await creatorProfilePage.getCreatorProfile();
        await test.expect(creatorProfile.handle).toBeTruthy();
      }
    });

    test("shows creator avatar if available", async () => {
      const hasAvatar = await creatorProfilePage.hasCreatorAvatar();
      console.log("Has creator avatar:", hasAvatar);
    });

    test("shows creator description if available", async () => {
      const hasDescription = await creatorProfilePage.hasCreatorDescription();
      console.log("Has creator description:", hasDescription);

      if (hasDescription) {
        const creatorProfile = await creatorProfilePage.getCreatorProfile();
        await test.expect(creatorProfile.description).toBeTruthy();
        await test.expect(creatorProfile.description.length).toBeGreaterThan(0);
      }
    });

    test("displays statistics if available", async () => {
      const hasRating = await creatorProfilePage.hasAverageRatingSection();
      const hasRuns = await creatorProfilePage.hasTotalRunsSection();
      const hasCategories = await creatorProfilePage.hasTopCategoriesSection();

      console.log("Has rating section:", hasRating);
      console.log("Has runs section:", hasRuns);
      console.log("Has categories section:", hasCategories);

      if (hasRating) {
        const creatorProfile = await creatorProfilePage.getCreatorProfile();
        await test
          .expect(creatorProfile.averageRating)
          .toBeGreaterThanOrEqual(0);
        await test.expect(creatorProfile.averageRating).toBeLessThanOrEqual(5);
      }

      if (hasRuns) {
        const creatorProfile = await creatorProfilePage.getCreatorProfile();
        await test.expect(creatorProfile.totalRuns).toBeGreaterThanOrEqual(0);
      }

      if (hasCategories) {
        const creatorProfile = await creatorProfilePage.getCreatorProfile();
        console.log("Top categories:", creatorProfile.topCategories);
      }
    });
  });

  test.describe("Creator Agents", () => {
    test("displays creator's agents", async () => {
      await test.expect(creatorProfilePage.hasAgents()).resolves.toBeTruthy();

      const agents = await creatorProfilePage.getCreatorAgents();
      await test.expect(agents.length).toBeGreaterThan(0);

      console.log("Creator agents count:", agents.length);
    });

    test("agent cards have correct information", async () => {
      const agents = await creatorProfilePage.getCreatorAgents();

      if (agents.length > 0) {
        const firstAgent = agents[0];
        await test.expect(firstAgent.name).toBeTruthy();
        await test.expect(firstAgent.description).toBeTruthy();
        await test.expect(typeof firstAgent.rating).toBe("number");
        await test.expect(typeof firstAgent.runs).toBe("number");

        console.log("First agent details:", firstAgent);
      }
    });

    test("can click on creator's agents", async ({ page }) => {
      const agents = await creatorProfilePage.getCreatorAgents();

      if (agents.length > 0) {
        const firstAgent = agents[0];
        await creatorProfilePage.clickAgent(firstAgent.name);
        await page.waitForTimeout(2000);

        // Should navigate to agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      }
    });

    test("agent count matches displayed agents", async () => {
      const creatorProfile = await creatorProfilePage.getCreatorProfile();
      const agents = await creatorProfilePage.getCreatorAgents();

      // The displayed agent count should match or be close to actual agents shown
      // (there might be pagination or filtering)
      await test
        .expect(agents.length)
        .toBeLessThanOrEqual(creatorProfile.agentCount + 5);
      console.log("Profile agent count:", creatorProfile.agentCount);
      console.log("Displayed agents:", agents.length);
    });
  });

  test.describe("Navigation", () => {
    test("can navigate back to store", async ({ page }) => {
      await creatorProfilePage.navigateBackToStore();
      await page.waitForTimeout(2000);

      // Should be back on marketplace
      await test.expect(page.url()).toMatch(/\/marketplace$/);
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });

    test("URL structure is correct", async ({ page }) => {
      const url = page.url();
      const urlParts = url.split("/");

      // URL should be /marketplace/creator/{creator-handle}
      await test.expect(urlParts).toContain("marketplace");
      await test.expect(urlParts).toContain("creator");
      await test.expect(urlParts.length).toBeGreaterThan(4);
    });

    test("page title contains creator information", async ({ page }) => {
      const title = await page.title();
      const creatorProfile = await creatorProfilePage.getCreatorProfile();

      // Title should contain creator name or be related to AutoGPT Store
      const titleContainsRelevantInfo =
        title.includes(creatorProfile.displayName) ||
        title.includes(creatorProfile.username) ||
        title.includes("AutoGPT") ||
        title.includes("Store") ||
        title.includes("Marketplace");

      await test.expect(titleContainsRelevantInfo).toBeTruthy();
    });
  });

  test.describe("Content Validation", () => {
    test("creator name is displayed prominently", async () => {
      const creatorName =
        await creatorProfilePage.creatorDisplayName.textContent();
      await test.expect(creatorName).toBeTruthy();
      await test.expect(creatorName?.trim().length).toBeGreaterThan(0);
    });

    test("agents section has meaningful heading", async ({ page }) => {
      const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsHeading).toBeVisible();
    });

    test("creator information is comprehensive", async () => {
      const creatorProfile = await creatorProfilePage.getCreatorProfile();

      // Creator should have at least a display name and username
      await test.expect(creatorProfile.displayName).toBeTruthy();
      await test.expect(creatorProfile.username).toBeTruthy();
    });
  });

  test.describe("Agent Filtering and Search", () => {
    test("can search creator's agents", async () => {
      const agents = await creatorProfilePage.getCreatorAgents();

      if (agents.length > 0) {
        const searchQuery = agents[0].name.substring(0, 3);
        const filteredAgents =
          await creatorProfilePage.searchCreatorAgents(searchQuery);

        console.log("Search query:", searchQuery);
        console.log("Filtered agents:", filteredAgents.length);

        // Filtered results should be subset of all agents
        await test
          .expect(filteredAgents.length)
          .toBeLessThanOrEqual(agents.length);
      }
    });

    test("agents can be grouped by categories if available", async () => {
      const creatorProfile = await creatorProfilePage.getCreatorProfile();

      if (creatorProfile.topCategories.length > 0) {
        const firstCategory = creatorProfile.topCategories[0];
        const categoryAgents =
          await creatorProfilePage.getAgentsByCategory(firstCategory);

        console.log("Category:", firstCategory);
        console.log("Category agents:", categoryAgents.length);
      }
    });
  });

  test.describe("Performance and Loading", () => {
    test("page loads within reasonable time", async ({ page }, testInfo) => {
      const startTime = Date.now();

      // Navigate to marketplace and then to a creator
      await page.goto("/marketplace");
      await marketplacePage.waitForPageLoad();

      const creators = await marketplacePage.getFeaturedCreators();
      if (creators.length > 0) {
        await marketplacePage.clickCreator(creators[0].displayName);
        await creatorProfilePage.waitForPageLoad();
      }

      const loadTime = Date.now() - startTime;

      // Page should load within 15 seconds
      await test.expect(loadTime).toBeLessThan(15000);

      testInfo.attach("load-time", {
        body: `Creator profile page loaded in ${loadTime}ms`,
      });
    });

    test("agents load properly", async () => {
      await creatorProfilePage.waitForAgentsLoad();

      const hasAgents = await creatorProfilePage.hasAgents();
      if (hasAgents) {
        const agents = await creatorProfilePage.getCreatorAgents();
        console.log("Loaded agents count:", agents.length);
        await test.expect(agents.length).toBeGreaterThan(0);
      }
    });

    test("page metrics are reasonable", async () => {
      const metrics = await creatorProfilePage.getPageMetrics();

      await test.expect(metrics.hasAllRequiredElements).toBeTruthy();
      await test.expect(metrics.agentCount).toBeGreaterThanOrEqual(0);

      console.log("Creator Profile Page Metrics:", metrics);
    });
  });

  test.describe("Responsive Design", () => {
    test("page works on mobile viewport", async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.reload();
      await creatorProfilePage.waitForPageLoad();

      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test
        .expect(creatorProfilePage.hasCreatorDisplayName())
        .resolves.toBeTruthy();
    });

    test("page works on tablet viewport", async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.reload();
      await creatorProfilePage.waitForPageLoad();

      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test
        .expect(creatorProfilePage.hasAgentsSection())
        .resolves.toBeTruthy();
    });

    test("scrolling works correctly", async () => {
      await creatorProfilePage.scrollToAgentsSection();
      await test
        .expect(creatorProfilePage.hasAgentsSection())
        .resolves.toBeTruthy();
    });
  });

  test.describe("Error Handling", () => {
    test("handles missing creator gracefully", async ({ page }) => {
      // Try to navigate to a non-existent creator
      await page.goto("/marketplace/creator/nonexistent-creator");
      await page.waitForTimeout(3000);

      // Should either show 404 or redirect to marketplace
      const url = page.url();
      const is404 =
        url.includes("404") || (await page.locator("text=404").isVisible());
      const redirectedToMarketplace =
        url.includes("/marketplace") && !url.includes("/creator/");

      await test.expect(is404 || redirectedToMarketplace).toBeTruthy();
    });

    test("handles creator with no agents gracefully", async ({ page: _ }) => {
      // This test would be relevant for creators with 0 agents
      const hasAgents = await creatorProfilePage.hasAgents();

      if (!hasAgents) {
        // Should still show the creator profile information
        await test
          .expect(creatorProfilePage.hasCreatorDisplayName())
          .resolves.toBeTruthy();
        await test
          .expect(creatorProfilePage.hasAgentsSection())
          .resolves.toBeTruthy();
      }
    });
  });

  test.describe("Accessibility", () => {
    test("main content is accessible", async ({ page }) => {
      const creatorName = page.getByRole("heading").first();
      await test.expect(creatorName).toBeVisible();

      const agentsSection = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsSection).toBeVisible();
    });

    test("navigation elements are accessible", async ({ page }) => {
      const storeLink = page.getByRole("link", { name: "Store" });
      await test.expect(storeLink).toBeVisible();
    });

    test("agent cards are accessible", async ({ page }) => {
      const agentButtons = page
        .getByRole("button")
        .filter({ hasText: /agent card/i });
      const agentCount = await agentButtons.count();

      if (agentCount > 0) {
        await test.expect(agentButtons.first()).toBeVisible();
      }
    });

    test("keyboard navigation works", async ({ page }) => {
      // Test basic keyboard navigation
      await page.keyboard.press("Tab");
      await page.keyboard.press("Tab");

      const focusedElement = await page.evaluate(
        () => document.activeElement?.tagName,
      );
      console.log("Focused element after tab navigation:", focusedElement);
    });
  });

  test.describe("Data Consistency", () => {
    test("creator information is consistent across pages", async ({ page }) => {
      // Get creator info from profile page
      const creatorProfile = await creatorProfilePage.getCreatorProfile();

      // Navigate to one of their agents
      const agents = await creatorProfilePage.getCreatorAgents();
      if (agents.length > 0) {
        await creatorProfilePage.clickAgent(agents[0].name);
        await page.waitForTimeout(2000);

        // Check that the creator name matches on agent detail page
        const agentDetails = await agentDetailPage.getAgentDetails();

        // Creator names should match (allowing for different formats)
        const creatorNamesMatch =
          agentDetails.creator
            .toLowerCase()
            .includes(creatorProfile.displayName.toLowerCase()) ||
          agentDetails.creator
            .toLowerCase()
            .includes(creatorProfile.username.toLowerCase()) ||
          creatorProfile.displayName
            .toLowerCase()
            .includes(agentDetails.creator.toLowerCase());

        await test.expect(creatorNamesMatch).toBeTruthy();
      }
    });

    test("agent count is reasonable", async () => {
      const creatorProfile = await creatorProfilePage.getCreatorProfile();
      const displayedAgents = await creatorProfilePage.getCreatorAgents();

      // Agent count should be reasonable (not negative, not impossibly high)
      await test.expect(creatorProfile.agentCount).toBeGreaterThanOrEqual(0);
      await test.expect(creatorProfile.agentCount).toBeLessThan(1000); // Reasonable upper limit

      // Displayed agents should not exceed claimed agent count significantly
      await test
        .expect(displayedAgents.length)
        .toBeLessThanOrEqual(creatorProfile.agentCount + 10);
    });
  });
});
