import { Page } from "@playwright/test";
import { BasePage } from "./base.page";
import { getSelectors } from "../utils/selectors";

export class MarketplacePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async goto() {
    await this.page.goto("/marketplace");
  }

  async getMarketplaceTitle() {
    const { getText } = getSelectors(this.page);
    return getText("Explore AI agents", { exact: false });
  }

  async getCreatorsSection() {
    const { getId, getText } = getSelectors(this.page);
    return getId("creators-section") || getText("Creators", { exact: false });
  }

  async getAgentsSection() {
    const { getId, getText } = getSelectors(this.page);
    return getId("agents-section") || getText("Agents", { exact: false });
  }

  async getCreatorsLink() {
    const { getLink } = getSelectors(this.page);
    return getLink(/creators/i);
  }

  async getAgentsLink() {
    const { getLink } = getSelectors(this.page);
    return getLink(/agents/i);
  }

  async getSearchInput() {
    const { getField, getId } = getSelectors(this.page);
    return getId("store-search-input") || getField(/search/i);
  }

  async getFilterDropdown() {
    const { getId, getButton } = getSelectors(this.page);
    return getId("filter-dropdown") || getButton(/filter/i);
  }

  async searchFor(query: string) {
    const searchInput = await this.getSearchInput();
    await searchInput.fill(query);
    await searchInput.press("Enter");
  }

  async clickCreators() {
    const creatorsLink = await this.getCreatorsLink();
    await creatorsLink.click();
  }

  async clickAgents() {
    const agentsLink = await this.getAgentsLink();
    await agentsLink.click();
  }

  async openFilter() {
    const filterDropdown = await this.getFilterDropdown();
    await filterDropdown.click();
  }

  async getFeaturedAgentsSection() {
    const { getText } = getSelectors(this.page);
    return getText("Featured agents");
  }

  async getTopAgentsSection() {
    const { getText } = getSelectors(this.page);
    return getText("Top Agents");
  }

  async getFeaturedCreatorsSection() {
    const { getText } = getSelectors(this.page);
    return getText("Featured Creators");
  }

  async getFeaturedAgentCards() {
    const { getId } = getSelectors(this.page);
    return getId("featured-store-card");
  }

  async getTopAgentCards() {
    const { getId } = getSelectors(this.page);
    return getId("store-card");
  }

  async getCreatorProfiles() {
    const { getId } = getSelectors(this.page);
    return getId("creator-card");
  }

  async searchAndNavigate(query: string) {
    const searchInput = await this.getSearchInput();
    await searchInput.fill(query);
    await searchInput.press("Enter");
  }

  async waitForSearchResults() {
    await this.page.waitForURL("**/marketplace/search**");
  }

  async getFirstFeaturedAgent() {
    const { getId } = getSelectors(this.page);
    return getId("featured-store-card").first();
  }

  async getFirstTopAgent() {
    return this.page.locator('[data-testid="store-card"]:visible').first();
  }

  async getFirstCreatorProfile() {
    const { getId } = getSelectors(this.page);
    return getId("creator-card").first();
  }

  async getSearchResultsCount() {
    const { getId } = getSelectors(this.page);
    const storeCards = getId("store-card");
    return await storeCards.count();
  }
}
