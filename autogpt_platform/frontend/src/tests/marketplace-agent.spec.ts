import { expect, test } from "@playwright/test";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { MarketplacePage } from "./pages/marketplace.page";
import { hasUrl, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

test.describe("Marketplace Agent Page - Cross-Page Flows", () => {
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
