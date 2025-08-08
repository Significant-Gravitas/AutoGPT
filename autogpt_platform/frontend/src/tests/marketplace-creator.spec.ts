import { test } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { LoginPage } from "./pages/login.page";
import { isVisible, hasUrl, matchesUrl } from "./utils/assertion";
import { TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";

test.describe("Marketplace Creator Page – Basic Functionality", () => {
  test("User can access creator's page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();

    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);
  });

  test("User can access creator's page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();

    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);
  });

  test("Creator page details are visible", async ({ page }) => {
    const { getId } = getSelectors(page);
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");

    const creatorTitle = getId("creator-title");
    await isVisible(creatorTitle);

    const creatorDescription = getId("creator-description");
    await isVisible(creatorDescription);
  });

  test("Agents in agent by sections navigation works", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");
    const firstAgent = page
      .locator('[data-testid="store-card"]:visible')
      .first();

    await firstAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
  });
});
