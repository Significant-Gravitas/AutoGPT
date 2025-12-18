import { Page } from "@playwright/test";
import { BasePage } from "./base.page";
import { getSelectors } from "../utils/selectors";

export class MarketplacePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async goto(page: Page) {
    await page.goto("/marketplace");
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
    const { getField, getId } = getSelectors(page);
    return getId("store-search-input") || getField(/search/i);
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
    return getText("Top Agents");
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
    return getId("featured-store-card").first();
  }

  async getFirstTopAgent() {
    return this.page.locator('[data-testid="store-card"]:visible').first();
  }

  async getFirstCreatorProfile(page: Page) {
    const { getId } = getSelectors(page);
    return getId("creator-card").first();
  }

  async getSearchResultsCount(page: Page) {
    const { getId } = getSelectors(page);
    const storeCards = getId("store-card");
    return await storeCards.count();
  }
}
