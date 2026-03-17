import { expect, test } from "@playwright/test";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { MarketplacePage } from "./pages/marketplace.page";
import { hasUrl, isVisible, matchesUrl } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

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
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
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
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
    await hasUrl(page, "/marketplace");
    await marketplacePage.goto(page);

    const firstStoreCard = await marketplacePage.getFirstTopAgent();
    await firstStoreCard.click();
    await page.waitForURL("**/marketplace/agent/**");

    const agentTitle = await getId("agent-title").textContent();
    if (!agentTitle || !agentTitle.trim()) {
      throw new Error("Agent title not found on marketplace agent page");
    }
    const agentName = agentTitle.trim();

    const addToLibraryButton = getId("agent-add-library-button");
    await isVisible(addToLibraryButton);
    await addToLibraryButton.click();

    const addSuccessMessage = getText("Redirecting to your library...");
    await isVisible(addSuccessMessage);

    await page.waitForURL("**/library/agents/**");
    await expect(page).toHaveTitle(
      new RegExp(`${escapeRegExp(agentName)} - Library - AutoGPT Platform`),
    );
  });
});
