import { expect, Page } from "@playwright/test";
import { BasePage } from "./base.page";
import { dismissFeedbackDialog } from "./library.page";
import { getSelectors } from "../utils/selectors";

const DETERMINISTIC_MARKETPLACE_AGENT_SEARCH = "E2E Calculator Agent";

export class MarketplacePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async goto(page: Page) {
    await page.goto("/marketplace");
    await page
      .locator(
        '[data-testid="store-card"], [data-testid="featured-store-card"]',
      )
      .first()
      .waitFor({ state: "visible", timeout: 20000 });
  }

  async getMarketplaceTitle(page: Page) {
    const { getText } = getSelectors(page);
    return getText("Explore AI agents", { exact: false });
  }

  async getCreatorsSection(page: Page) {
    const { getId, getText } = getSelectors(page);
    return getId("creators-section") || getText("Creators", { exact: false });
  }

  async getAgentsSection(page: Page) {
    const { getId, getText } = getSelectors(page);
    return getId("agents-section") || getText("Agents", { exact: false });
  }

  async getCreatorsLink(page: Page) {
    const { getLink } = getSelectors(page);
    return getLink(/creators/i);
  }

  async getAgentsLink(page: Page) {
    const { getLink } = getSelectors(page);
    return getLink(/agents/i);
  }

  async getSearchInput(page: Page) {
    const visibleSearchInput = page
      .locator('[data-testid="store-search-input"]:visible')
      .first();
    if (await visibleSearchInput.isVisible().catch(() => false)) {
      return visibleSearchInput;
    }

    const { getField, getId } = getSelectors(page);
    return getId("store-search-input").first() || getField(/search/i).first();
  }

  async getFilterDropdown(page: Page) {
    const { getId, getButton } = getSelectors(page);
    return getId("filter-dropdown") || getButton(/filter/i);
  }

  async searchFor(query: string, page: Page) {
    const searchInput = await this.getSearchInput(page);
    await searchInput.fill(query);
    await searchInput.press("Enter");
  }

  async clickCreators(page: Page) {
    const creatorsLink = await this.getCreatorsLink(page);
    await creatorsLink.click();
  }

  async clickAgents(page: Page) {
    const agentsLink = await this.getAgentsLink(page);
    await agentsLink.click();
  }

  async openFilter(page: Page) {
    const filterDropdown = await this.getFilterDropdown(page);
    await filterDropdown.click();
  }

  async getFeaturedAgentsSection(page: Page) {
    const { getText } = getSelectors(page);
    return getText("Featured agents");
  }

  async getTopAgentsSection(page: Page) {
    const { getText } = getSelectors(page);
    return getText("All Agents");
  }

  async getFeaturedCreatorsSection(page: Page) {
    const { getText } = getSelectors(page);
    return getText("Featured Creators");
  }

  async getFeaturedAgentCards(page: Page) {
    const { getId } = getSelectors(page);
    return getId("featured-store-card");
  }

  async getTopAgentCards(page: Page) {
    const { getId } = getSelectors(page);
    return getId("store-card");
  }

  async getCreatorProfiles(page: Page) {
    const { getId } = getSelectors(page);
    return getId("creator-card");
  }

  async searchAndNavigate(query: string, page: Page) {
    const searchInput = (await this.getSearchInput(page)).first();
    await searchInput.fill(query);
    await searchInput.press("Enter");
  }

  async waitForSearchResults() {
    await this.page.waitForURL("**/marketplace/search**");
  }

  async getFirstFeaturedAgent(page: Page) {
    const { getId } = getSelectors(page);
    const card = getId("featured-store-card").first();
    await card.waitFor({ state: "visible", timeout: 15000 });
    return card;
  }

  async getFirstTopAgent() {
    const card = this.page
      .locator('[data-testid="store-card"]:visible')
      .first();
    await card.waitFor({ state: "visible", timeout: 15000 });
    return card;
  }

  async getFirstCreatorProfile(page: Page) {
    const { getId } = getSelectors(page);
    const card = getId("creator-card").first();
    await card.waitFor({ state: "visible", timeout: 15000 });
    return card;
  }

  async getSearchResultsCount(page: Page) {
    const { getId } = getSelectors(page);
    const storeCards = getId("store-card");
    return await storeCards.count();
  }

  // --- Happy-path flows shared across PR smoke specs ---

  async openRunnableAgent(): Promise<{ path: string }> {
    await this.searchAndOpenAgent(DETERMINISTIC_MARKETPLACE_AGENT_SEARCH);

    await expect(this.page.getByTestId("agent-add-library-button")).toBeVisible(
      {
        timeout: 15000,
      },
    );

    return { path: this.page.url() };
  }

  async openFeaturedAgent(): Promise<void> {
    await this.searchAndOpenAgent(DETERMINISTIC_MARKETPLACE_AGENT_SEARCH);
    await dismissFeedbackDialog(this.page);
  }

  private async searchAndOpenAgent(agentName: string): Promise<void> {
    const searchURL = `/marketplace/search?searchTerm=${encodeURIComponent(agentName)}`;

    const agentCard = this.page
      .locator('[data-testid="store-card"]:visible')
      .filter({ hasText: agentName })
      .first();

    for (let attempt = 0; attempt < 3; attempt++) {
      await this.page.goto(searchURL);
      await this.page.waitForLoadState("networkidle");

      const visible = await agentCard
        .waitFor({ state: "visible", timeout: 15000 })
        .then(() => true)
        .catch(() => false);

      if (visible) break;

      if (attempt === 2) {
        await expect(agentCard).toBeVisible({ timeout: 15000 });
      }
    }

    await agentCard.click();

    await expect(this.page).toHaveURL(/\/marketplace\/agent\//, {
      timeout: 15000,
    });
    await expect(this.page.getByTestId("agent-title")).toBeVisible({
      timeout: 15000,
    });
  }

  async submitAgentForReview(publishableAgentName: string): Promise<{
    agentTitle: string;
    agentSlug: string;
  }> {
    await this.page.goto("/marketplace");
    await this.page.getByRole("button", { name: "Become a Creator" }).click();

    const publishAgentModal = this.page.getByTestId("publish-agent-modal");
    await expect(publishAgentModal).toBeVisible();
    await expect(
      publishAgentModal.getByText(
        "Select your project that you'd like to publish",
      ),
    ).toBeVisible();

    const publishableAgentCard = publishAgentModal
      .getByTestId("agent-card")
      .filter({ hasText: publishableAgentName })
      .first();
    await expect(publishableAgentCard).toBeVisible({ timeout: 15000 });
    await publishableAgentCard.click();
    await publishAgentModal
      .getByRole("button", { name: "Next", exact: true })
      .click();

    await expect(
      publishAgentModal.getByText("Write a bit of details about your agent"),
    ).toBeVisible();

    const suffix = Date.now().toString().slice(-6);
    const agentTitle = `Publish Flow ${suffix}`;
    const agentSlug = `publish-flow-${suffix}`;

    await publishAgentModal.getByLabel("Title").fill(agentTitle);
    await publishAgentModal
      .getByLabel("Subheader")
      .fill("A deterministic marketplace submission");
    await publishAgentModal.getByLabel("Slug").fill(agentSlug);
    await publishAgentModal
      .getByLabel("YouTube video link")
      .fill("https://www.youtube.com/watch?v=test123");

    await publishAgentModal.getByRole("combobox", { name: "Category" }).click();
    await this.page.getByRole("option", { name: "Other" }).click();

    await publishAgentModal
      .getByLabel("Description")
      .fill(
        "A deterministic publish flow for consolidated Playwright coverage.",
      );

    const submitButton = publishAgentModal.getByRole("button", {
      name: "Submit for review",
    });
    await expect(submitButton).toBeEnabled();
    await submitButton.click();

    await expect(
      publishAgentModal.getByText("Agent is awaiting review"),
    ).toBeVisible();
    await expect(
      publishAgentModal.getByTestId("view-progress-button"),
    ).toBeVisible();

    return { agentTitle, agentSlug };
  }

  async waitForDashboardSubmission(agentTitle: string) {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const submissionRow = this.page
        .getByTestId("agent-table-row")
        .filter({ hasText: agentTitle })
        .first();

      // Row may not appear immediately after redirect — allow a short render
      // window before deciding the submission is absent on this attempt.
      if (await submissionRow.isVisible({ timeout: 5000 }).catch(() => false)) {
        return submissionRow;
      }

      await this.page.reload();
      await expect(this.page).toHaveURL(/\/profile\/dashboard/);
      await expect(this.page.getByText("Agent dashboard")).toBeVisible();
    }

    throw new Error(`Submission row for "${agentTitle}" did not appear`);
  }
}
