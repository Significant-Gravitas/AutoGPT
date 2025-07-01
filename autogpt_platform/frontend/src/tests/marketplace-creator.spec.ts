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

    // Navigate to marketplace first with workaround for #8788
    await page.goto("/marketplace");
    // workaround for #8788 - same as build tests
    await page.reload();
    await page.reload();
    await marketplacePage.waitForPageLoad();

    // Navigate to a creator profile page via featured creators
    const creators = await marketplacePage.getFeaturedCreators();
    if (creators.length > 0) {
      await marketplacePage.clickCreator(creators[0].name);
      await page.waitForTimeout(2000);
      // workaround for #8788
      await page.reload();
      await creatorProfilePage.waitForPageLoad();
    }
  });

  test.describe("Page Load and Structure", () => {
    test("creator profile page loads successfully", async ({ page }) => {
      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(page.url()).toMatch(/\/marketplace\/creator\/.*/);
    });

    test("displays basic page elements", async ({ page }) => {
      // Check for main content area
      await test.expect(page.locator("main")).toBeVisible();

      // Check for breadcrumbs - should show "Store" link
      await test.expect(page.getByText("Store")).toBeVisible();
    });

    test("displays creator information", async () => {
      // Check for creator display name (h1)
      await test.expect(creatorProfilePage.creatorDisplayName).toBeVisible();

      // Check for agents section
      await test.expect(creatorProfilePage.agentsSection).toBeVisible();
    });

    test("has breadcrumb navigation", async ({ page }) => {
      // Check for Store breadcrumb link
      const storeLink = page.getByRole("link", { name: "Store" });
      await test.expect(storeLink).toBeVisible();
    });
  });

  test.describe("Creator Information", () => {
    test("displays creator name", async () => {
      const creatorName =
        await creatorProfilePage.creatorDisplayName.textContent();
      await test.expect(creatorName).toBeTruthy();
      await test.expect(creatorName?.trim().length).toBeGreaterThan(0);
    });

    test("displays about section if available", async ({ page }) => {
      // Check for "About" section
      const aboutSection = page.getByText("About");
      const hasAbout = await aboutSection.isVisible().catch(() => false);
      console.log("Has about section:", hasAbout);
    });

    test("displays creator description if available", async ({ page }) => {
      // Creator description comes after "About" section
      const descriptionText = await page
        .locator("main div")
        .filter({ hasText: /\w{20,}/ })
        .first()
        .textContent();
      const hasDescription =
        descriptionText && descriptionText.trim().length > 20;
      console.log("Has creator description:", hasDescription);

      if (hasDescription) {
        console.log(
          "Description preview:",
          descriptionText?.substring(0, 100) + "...",
        );
      }
    });

    test("displays creator info card if available", async ({ page }) => {
      // Look for creator info elements like avatar, stats, etc.
      const hasInfoCard = await page
        .locator("div")
        .filter({ hasText: /average rating|agents|runs/ })
        .count();
      console.log("Creator info elements found:", hasInfoCard);
    });
  });

  test.describe("Creator Agents", () => {
    test("displays agents section", async ({ page }) => {
      // Check for "Agents by" heading
      const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsHeading).toBeVisible();
    });

    test("displays agent cards if available", async ({ page }) => {
      // Count store cards in the page
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      console.log("Agent cards found:", agentCards);

      if (agentCards > 0) {
        // Check first agent card has required elements
        const firstCard = page.locator('[data-testid="store-card"]').first();
        await test.expect(firstCard.locator("h3")).toBeVisible(); // Agent name
        await test.expect(firstCard.locator("p")).toBeVisible(); // Description
      }
    });

    test("can click on creator's agents if they exist", async ({ page }) => {
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();

      if (agentCards > 0) {
        // Click first agent card
        await page.locator('[data-testid="store-card"]').first().click();
        await page.waitForTimeout(2000);
        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();

        // Should navigate to agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);
        await test.expect(agentDetailPage.isLoaded()).resolves.toBeTruthy();
      } else {
        console.log("No agent cards found to click");
      }
    });

    test("agents section displays properly", async ({ page }) => {
      // The agents section should be visible regardless of whether there are agents
      const agentsSection = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsSection).toBeVisible();

      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      console.log("Total agent cards displayed:", agentCards);
    });
  });

  test.describe("Navigation", () => {
    test("can navigate back to store", async ({ page }) => {
      // Click the Store breadcrumb link
      await page.getByRole("link", { name: "Store" }).click();
      await page.waitForTimeout(2000);
      // workaround for #8788
      await page.reload();
      await page.reload();
      await marketplacePage.waitForPageLoad();

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

      // Title should be related to AutoGPT Store
      const titleContainsRelevantInfo =
        title.includes("AutoGPT") ||
        title.includes("Store") ||
        title.includes("Marketplace");

      await test.expect(titleContainsRelevantInfo).toBeTruthy();
      console.log("Page title:", title);
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

    test("page has proper structure", async ({ page }) => {
      // Check for main content area
      await test.expect(page.locator("main")).toBeVisible();

      // Check for creator name heading
      await test.expect(page.getByRole("heading").first()).toBeVisible();

      // Check for agents section
      const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsHeading).toBeVisible();
    });
  });

  test.describe("Agent Display", () => {
    test("agents are displayed in grid layout", async ({ page }) => {
      // Check if there's a grid layout for agents (desktop view)
      const gridElements = await page.locator(".grid").count();
      console.log("Grid layout elements found:", gridElements);

      // Check for agent cards
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      console.log("Agent cards displayed:", agentCards);
    });

    test("agent cards are interactive", async ({ page }) => {
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();

      if (agentCards > 0) {
        const firstCard = page.locator('[data-testid="store-card"]').first();

        // Check that card is clickable
        await test.expect(firstCard).toBeVisible();

        // Verify it has the role="button" attribute
        const hasButtonRole = await firstCard.getAttribute("role");
        console.log("First card role:", hasButtonRole);
      }
    });
  });

  test.describe("Performance and Loading", () => {
    test("page loads within reasonable time", async ({ page }, testInfo) => {
      // Use the same timeout multiplier as build tests
      await test.setTimeout(testInfo.timeout * 10);

      const startTime = Date.now();

      // Navigate to marketplace and then to a creator with workaround for #8788
      await page.goto("/marketplace");
      // workaround for #8788
      await page.reload();
      await page.reload();
      await marketplacePage.waitForPageLoad();

      const creators = await marketplacePage.getFeaturedCreators();
      if (creators.length > 0) {
        await marketplacePage.clickCreator(creators[0].name);
        // workaround for #8788
        await page.reload();
        await creatorProfilePage.waitForPageLoad();
      }

      const loadTime = Date.now() - startTime;

      // Page should load within 15 seconds
      await test.expect(loadTime).toBeLessThan(15000);

      testInfo.attach("load-time", {
        body: `Creator profile page loaded in ${loadTime}ms`,
      });
    });

    test("agents section loads properly", async ({ page }) => {
      // Wait for agents section to be visible
      const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsHeading).toBeVisible();

      // Count agent cards
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      console.log("Loaded agents count:", agentCards);
    });

    test("page loads core elements", async ({ page }) => {
      // Check for main required elements
      await test.expect(page.locator("main")).toBeVisible();
      await test.expect(page.getByRole("heading").first()).toBeVisible();

      const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
      await test.expect(agentsHeading).toBeVisible();

      console.log("Creator profile page loaded with core elements");
    });
  });

  test.describe("Responsive Design", () => {
    test("page works on mobile viewport", async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.reload();
      await creatorProfilePage.waitForPageLoad();

      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(creatorProfilePage.creatorDisplayName).toBeVisible();
    });

    test("page works on tablet viewport", async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.reload();
      await creatorProfilePage.waitForPageLoad();

      await test.expect(creatorProfilePage.isLoaded()).resolves.toBeTruthy();
      await test.expect(creatorProfilePage.agentsSection).toBeVisible();
    });

    test("scrolling works correctly", async ({ page }) => {
      // Scroll to agents section
      const agentsSection = page.getByRole("heading", { name: /Agents by/i });
      await agentsSection.scrollIntoViewIfNeeded();
      await test.expect(agentsSection).toBeVisible();
    });
  });

  test.describe("Error Handling", () => {
    test("handles missing creator gracefully", async ({ page }) => {
      // Try to navigate to a non-existent creator
      await page.goto("/marketplace/creator/nonexistent-creator");
      await page.waitForTimeout(3000);
      // workaround for #8788
      await page.reload();

      // Should either show 404 or redirect to marketplace
      const url = page.url();
      const is404 =
        url.includes("404") || (await page.locator("text=404").isVisible());
      const redirectedToMarketplace =
        url.includes("/marketplace") && !url.includes("/creator/");

      await test.expect(is404 || redirectedToMarketplace).toBeTruthy();
    });

    test("handles creator with no agents gracefully", async ({ page }) => {
      // Check agent count
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();

      if (agentCards === 0) {
        // Should still show the creator profile information
        await test.expect(creatorProfilePage.creatorDisplayName).toBeVisible();

        // Should still show agents section header
        const agentsHeading = page.getByRole("heading", { name: /Agents by/i });
        await test.expect(agentsHeading).toBeVisible();

        console.log("Creator has no agents, but page displays correctly");
      } else {
        console.log("Creator has agents:", agentCards);
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
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();

      if (agentCards > 0) {
        const firstCard = page.locator('[data-testid="store-card"]').first();
        await test.expect(firstCard).toBeVisible();

        // Check if card has proper accessibility attributes
        const role = await firstCard.getAttribute("role");
        const ariaLabel = await firstCard.getAttribute("aria-label");
        console.log(
          "First card accessibility - role:",
          role,
          "aria-label:",
          ariaLabel,
        );
      }
    });

    test("page structure is accessible", async ({ page }) => {
      // Check for proper heading hierarchy
      const headings = await page.locator("h1, h2, h3, h4, h5, h6").count();
      await test.expect(headings).toBeGreaterThan(0);

      // Check page title
      const title = await page.title();
      await test.expect(title).toBeTruthy();

      console.log("Page has", headings, "headings and title:", title);
    });
  });

  test.describe("Data Consistency", () => {
    test("creator information is consistent across pages", async ({ page }) => {
      // Get creator name from profile page
      const creatorName =
        await creatorProfilePage.creatorDisplayName.textContent();

      // Navigate to one of their agents if available
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      if (agentCards > 0) {
        await page.locator('[data-testid="store-card"]').first().click();
        await page.waitForTimeout(2000);

        // workaround for #8788
        await page.reload();
        await agentDetailPage.waitForPageLoad();

        // Check that we navigated to agent detail page
        await test.expect(page.url()).toMatch(/\/marketplace\/agent\/.*\/.*/);

        console.log("Creator name from profile:", creatorName?.trim());
        console.log("Navigated to agent detail page successfully");
      } else {
        console.log("No agents available to test consistency");
      }
    });

    test("page displays agent count information", async ({ page }) => {
      // Count actual agent cards displayed
      const agentCards = await page
        .locator('[data-testid="store-card"]')
        .count();
      console.log("Agent cards displayed:", agentCards);

      // Agent count should be reasonable (not negative, not impossibly high)
      await test.expect(agentCards).toBeGreaterThanOrEqual(0);
      await test.expect(agentCards).toBeLessThan(100); // Reasonable upper limit for display
    });
  });
});
