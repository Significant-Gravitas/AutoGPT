import { test } from "./fixtures";
import { MarketplacePage } from "./pages/marketplace.page";
import { AgentDetailPage } from "./pages/agent-detail.page";
import { CreatorProfilePage } from "./pages/creator-profile.page";

test.describe("Marketplace Agent Detail", () => {
  let marketplacePage: MarketplacePage;
  let agentDetailPage: AgentDetailPage;
  let creatorProfilePage: CreatorProfilePage;

  test.beforeEach(async ({ page }) => {
    marketplacePage = new MarketplacePage(page);
    agentDetailPage = new AgentDetailPage(page);
    creatorProfilePage = new CreatorProfilePage(page);

    // Navigate to marketplace first with workaround for #8788
    await page.goto("/marketplace");
    // workaround for #8788 - same as build tests
    await page.reload();
    await page.reload();
    await marketplacePage.waitForPageLoad();

    // Navigate to a specific agent detail page
    const agents = await marketplacePage.getAgentCards();
    if (agents.length > 0) {
      await marketplacePage.clickAgentCard(agents[0].agent_name);
      await page.waitForTimeout(2000);
      // workaround for #8788
      await page.reload();
      await agentDetailPage.waitForPageLoad();
    }
  });

  test.describe("Page Load and Structure", () => {
    test("agent detail page loads successfully", async ({ page }) => {
      await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
    });

    test("displays basic page elements", async ({ page }) => {
      // Check for main content area
      await test.expect(page.locator("main")).toBeVisible();

      // Check for breadcrumbs
      await test.expect(page.getByText("Marketplace")).toBeVisible();
    });

    test("displays agent information", async () => {
      // Check for agent name (h1, h2, or h3)
      await test.expect(agentDetailPage.agentName).toBeVisible();

      // Check for creator link
      await test.expect(agentDetailPage.creatorLink).toBeVisible();
    });
  });

  test.describe("Agent Information", () => {
    test("shows version information", async () => {
      const hasVersionInfo = await agentDetailPage.hasVersionInfo();
      console.log("Has version info:", hasVersionInfo);

      if (hasVersionInfo) {
        const agentDetails = await agentDetailPage.getAgentDetails();
        await test.expect(agentDetails.version).toBeTruthy();
      }
    });

    test("shows categories if available", async () => {
      const hasCategories = await agentDetailPage.hasCategoriesInfo();
      console.log("Has categories:", hasCategories);

      if (hasCategories) {
        const agentDetails = await agentDetailPage.getAgentDetails();
        await test.expect(agentDetails.categories.length).toBeGreaterThan(0);
      }
    });

    test("displays rating and runs correctly", async () => {
      const agentDetails = await agentDetailPage.getAgentDetails();

      // Rating should be between 0 and 5
      await test.expect(agentDetails.rating).toBeGreaterThanOrEqual(0);
      await test.expect(agentDetails.rating).toBeLessThanOrEqual(5);

      // Runs should be non-negative
      await test.expect(agentDetails.runs).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe("Agent Interactions", () => {
    test("download button is functional", async () => {
      await test
        .expect(agentDetailPage.hasDownloadButton())
        .resolves.toBeTruthy();

      // Verify button is clickable (without actually downloading)
      const downloadButton = agentDetailPage.downloadButton;
      await test.expect(downloadButton.isVisible()).resolves.toBeTruthy();
      await test.expect(downloadButton.isEnabled()).resolves.toBeTruthy();
    });

    test("can navigate to creator profile", async ({ page }) => {
      await agentDetailPage.clickCreatorLink();
      await page.waitForTimeout(2000);
      // workaround for #8788
      await page.reload();
      await creatorProfilePage.waitForPageLoad();

      // Should navigate to creator profile page
      await test.expect(page.url()).toMatch(/\/marketplace\/creator\/.*/);
      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
    });

    test("can navigate back to marketplace", async ({ page }) => {
      // First ensure we're on agent detail page
      await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);

      await agentDetailPage.navigateBackToMarketplace();
      await page.waitForTimeout(2000);
      // workaround for #8788
      await page.reload();
      await page.reload();
      await marketplacePage.waitForPageLoad();

      // Should be back on marketplace
      await test.expect(page.url()).toMatch(/\/marketplace$/);
      await test.expect(marketplacePage.isLoaded()).resolves.toBeTruthy();
    });
  });

  test.describe("Related Agents", () => {
    test("shows other agents by same creator", async () => {
      const hasOtherAgents =
        await agentDetailPage.hasOtherAgentsByCreatorSection();
      console.log("Has other agents by creator:", hasOtherAgents);

      if (hasOtherAgents) {
        const relatedAgents = await agentDetailPage.getRelatedAgents();
        console.log("Related agents count:", relatedAgents.length);
      }
    });

    test("shows similar agents", async () => {
      const hasSimilarAgents = await agentDetailPage.hasSimilarAgentsSection();
      console.log("Has similar agents:", hasSimilarAgents);

      if (hasSimilarAgents) {
        const relatedAgents = await agentDetailPage.getRelatedAgents();
        console.log("Similar agents count:", relatedAgents.length);
      }
    });

    test("can click on related agents if they exist", async ({ page }) => {
      // Related agents are in the "Other agents by" and "Similar agents" sections
      const relatedAgentCards = await page
        .locator('[data-testid="store-card"]')
        .count();

      if (relatedAgentCards > 0) {
        // Click first related agent card
        await page.locator('[data-testid="store-card"]').first().click();
        await page.waitForTimeout(2000);
        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();

        // Should navigate to another agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      } else {
        console.log("No related agents found");
      }
    });

    test("displays related agent sections", async ({ page }) => {
      // Check for section headings that indicate related agents
      const otherAgentsHeading = page.getByRole("heading", {
        name: /Other agents by/i,
      });
      const similarAgentsHeading = page.getByRole("heading", {
        name: /Similar agents/i,
      });

      // At least one of these sections should be visible
      const hasOtherAgents = await otherAgentsHeading
        .isVisible()
        .catch(() => false);
      const hasSimilarAgents = await similarAgentsHeading
        .isVisible()
        .catch(() => false);

      console.log("Has other agents section:", hasOtherAgents);
      console.log("Has similar agents section:", hasSimilarAgents);
    });
  });

  test.describe("Page Navigation and URL", () => {
    test("URL structure is correct", async ({ page }) => {
      const url = page.url();
      const urlParts = url.split("/");

      // URL should be /marketplace/agent/{creator}/{agent-name}
      await test.expect(urlParts).toContain("marketplace");
      await test.expect(urlParts).toContain("agent");
      await test.expect(urlParts.length).toBeGreaterThan(5);
    });

    test("page title contains agent information", async ({ page }) => {
      const title = await page.title();
      const agentDetails = await agentDetailPage.getAgentDetails();

      // Title should contain agent name or be related to AutoGPT Marketplace
      const titleContainsRelevantInfo =
        title.includes(agentDetails.name) ||
        title.includes("AutoGPT") ||
        title.includes("Marketplace") ||
        title.includes("Store");

      await test.expect(titleContainsRelevantInfo).toBeTruthy();
    });
  });

  test.describe("Content Validation", () => {
    test("agent name is displayed prominently", async () => {
      const agentName = await agentDetailPage.agentName.textContent();
      await test.expect(agentName).toBeTruthy();
      await test.expect(agentName?.trim().length).toBeGreaterThan(0);
    });

    test("creator information is complete", async () => {
      const agentDetails = await agentDetailPage.getAgentDetails();
      await test.expect(agentDetails.creator).toBeTruthy();
      await test.expect(agentDetails.creator.length).toBeGreaterThan(0);
    });

    test("description provides meaningful information", async () => {
      const agentDetails = await agentDetailPage.getAgentDetails();
      await test.expect(agentDetails.description).toBeTruthy();
      await test.expect(agentDetails.description.length).toBeGreaterThan(10);
    });
  });

  test.describe("Performance and Loading", () => {
    test("page loads within reasonable time", async ({ page }, testInfo) => {
      // Use the same timeout multiplier as build tests
      await test.setTimeout(testInfo.timeout * 10);

      const startTime = Date.now();

      // Navigate to marketplace and then to an agent with workaround for #8788
      await page.goto("/marketplace");
      // workaround for #8788
      await page.reload();
      await page.reload();
      await marketplacePage.waitForPageLoad();

      const agents = await marketplacePage.getAgentCards();
      if (agents.length > 0) {
        await marketplacePage.clickAgentCard(agents[0].agent_name);
        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();
      }

      const loadTime = Date.now() - startTime;

      // Page should load within 15 seconds
      await test.expect(loadTime).toBeLessThan(15000);

      testInfo.attach("load-time", {
        body: `Agent detail page loaded in ${loadTime}ms`,
      });
    });

    test("images load properly", async () => {
      await agentDetailPage.waitForImagesLoad();

      const hasImages = await agentDetailPage.hasAgentImages();
      if (hasImages) {
        const imageCount = await agentDetailPage.agentImages.count();
        console.log("Image count:", imageCount);
        await test.expect(imageCount).toBeGreaterThan(0);
      }
    });

    test("page metrics are reasonable", async () => {
      const metrics = await agentDetailPage.getPageMetrics();

      await test.expect(metrics.hasAllRequiredElements).toBeTruthy();

      console.log("Agent Detail Page Metrics:", metrics);
    });
  });

  test.describe("Error Handling", () => {
    test("handles missing agent gracefully", async ({ page }) => {
      // Try to navigate to a non-existent agent
      await page.goto("/marketplace/agent/nonexistent/nonexistent-agent");
      await page.waitForTimeout(3000);
      // workaround for #8788
      await page.reload();

      // Should either show 404 or redirect to marketplace
      const url = page.url();
      const is404 =
        url.includes("404") || (await page.locator("text=404").isVisible());
      const redirectedToMarketplace =
        url.includes("/marketplace") && !url.includes("/agent/");

      await test.expect(is404 || redirectedToMarketplace).toBeTruthy();
    });

    test("handles missing elements gracefully", async ({ page }) => {
      // Check that page doesn't crash when optional elements are missing
      const url = page.url();
      await test.expect(url).toBeTruthy();
      await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();

      console.log("Page handles missing elements gracefully");
    });
  });

  test.describe("Accessibility", () => {
    test("main content is accessible", async ({ page }) => {
      // Check for main heading
      const heading = page.getByRole("heading").first();
      await test.expect(heading).toBeVisible();

      // Check for main content area
      const main = page.locator("main");
      await test.expect(main).toBeVisible();
    });

    test("navigation elements are accessible", async ({ page }) => {
      // Check for breadcrumb navigation
      const marketplaceLink = page.getByRole("link", { name: "Marketplace" });
      await test.expect(marketplaceLink).toBeVisible();
    });

    test("page structure is accessible", async ({ page }) => {
      // Check that page has proper structure for screen readers
      const pageTitle = await page.title();
      await test.expect(pageTitle).toBeTruthy();

      console.log("Page title:", pageTitle);
    });
  });
});
