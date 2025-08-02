import { test, expect } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { LoginPage } from "./pages/login.page";
import { isVisible, hasUrl, matchesUrl } from "./utils/assertion";
import { TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";

test.describe("Marketplace Agent Page - Basic Functionality", () => {
  test("User can access agent page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();

    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
  });

  test("User can access agent page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");
    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();

    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
  });

  test("Agent page details are visible", async ({ page }) => {
    const { getId } = getSelectors(page);

    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();
    await page.waitForURL("**/marketplace/agent/**");

    const agentTitle = getId("agent-title");
    await isVisible(agentTitle);

    const agentDescription = getId("agent-description");
    await isVisible(agentDescription);

    const creatorInfo = getId("agent-creator");
    await isVisible(creatorInfo);
  });

  test("Download button functionality works", async ({ page }) => {
    const { getId, getText } = getSelectors(page);

    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();
    await page.waitForURL("**/marketplace/agent/**");

    const downloadButton = getId("agent-download-button");
    await isVisible(downloadButton);
    await downloadButton.click();

    const downloadSuccessMessage = getText(
      "Your agent has been successfully downloaded.",
    );
    await isVisible(downloadSuccessMessage);
  });

  test("Add to library button works and agent appears in library", async ({
    page,
  }) => {
    const { getId, getText } = getSelectors(page);

    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");
    await marketplacePage.goto(page);

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();
    await page.waitForURL("**/marketplace/agent/**");

    const agentName = await getId("agent-title").textContent();

    const addToLibraryButton = getId("agent-add-library-button");
    await isVisible(addToLibraryButton);
    await addToLibraryButton.click();

    const addSuccessMessage = getText("Redirecting to your library...");
    await isVisible(addSuccessMessage);

    await page.waitForURL("**/library/agents/**");
    const agentNameOnLibrary = await getId("agent-title").textContent();

    expect(agentNameOnLibrary?.trim()).toBe(agentName?.trim());
  });
});
