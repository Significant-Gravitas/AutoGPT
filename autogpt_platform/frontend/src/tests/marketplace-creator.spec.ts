import { test } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { hasUrl, matchesUrl } from "./utils/assertion";

test.describe("Marketplace Creator Page – Cross-Page Flows", () => {
  test("Agents in agent by sections navigation works", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");
    await page.waitForLoadState("networkidle").catch(() => {});

    const firstAgent = page
      .locator('[data-testid="store-card"]:visible')
      .first();
    await firstAgent.waitFor({ state: "visible", timeout: 30000 });

    await firstAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
  });
});
