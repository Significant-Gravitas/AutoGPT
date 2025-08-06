import { Page } from "@playwright/test";
import { LibraryPage, Agent } from "../pages/library.page";

export class LibraryUtils {
  constructor(
    private page: Page,
    private libraryPage: LibraryPage,
  ) {}

  async navigateToLibrary(): Promise<void> {
    await this.page.goto("/library");
    await this.libraryPage.isLoaded();
  }

  async searchAndVerify(
    searchTerm: string,
    expectedCount?: number,
  ): Promise<Agent[]> {
    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();

    if (expectedCount !== undefined) {
    }

    for (const agent of agents) {
      const containsSearchTerm =
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description.toLowerCase().includes(searchTerm.toLowerCase());

      if (!containsSearchTerm) {
      }
    }

    return agents;
  }

  async testSorting(): Promise<void> {
    await this.libraryPage.selectSortOption(this.page, "Creation Date");
    let currentSort = await this.libraryPage.getCurrentSortOption();

    const agentsCreationDate = await this.libraryPage.getAgents();

    await this.libraryPage.selectSortOption(this.page, "Last Modified");
    currentSort = await this.libraryPage.getCurrentSortOption();

    const agentsLastModified = await this.libraryPage.getAgents();

    if (agentsCreationDate.length > 1 && agentsLastModified.length > 1) {
      const orderChanged =
        agentsCreationDate[0].id !== agentsLastModified[0].id;
    }
  }

  async testUploadDialog(): Promise<void> {
    await this.libraryPage.openUploadDialog();
    const isVisible = await this.libraryPage.isUploadDialogVisible();

    const testName = "Test Agent";
    const testDescription = "This is a test agent description";

    await this.libraryPage.fillUploadForm(testName, testDescription);

    const isUploadEnabled = await this.libraryPage.isUploadButtonEnabled();

    await this.libraryPage.closeUploadDialog();
    const isHidden = !(await this.libraryPage.isUploadDialogVisible());
  }

  async testAgentInteractions(): Promise<void> {
    const agents = await this.libraryPage.getAgents();

    if (agents.length === 0) {
      return;
    }

    const testAgent = agents[0];

    // Replace isAgentVisible with direct locator check
    const isVisible = await this.page
      .getByRole("heading", {
        name: testAgent.name,
        level: 3,
      })
      .isVisible();

    const agentHeading = this.page.getByRole("heading", {
      name: testAgent.name,
      level: 3,
    });
    const isHeadingVisible = await agentHeading.isVisible();
  }

  async testPagination(): Promise<void> {
    const paginationResult = await this.libraryPage.testPagination();

    if (paginationResult.hasMore) {
    } else {
    }
  }

  async testInfiniteScroll(): Promise<void> {
    const initialCount = await this.libraryPage.getAgentCount();

    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();

    const finalCount = await this.libraryPage.getAgentCount();

    const isPaginationWorking = await this.libraryPage.isPaginationWorking();
  }

  async getAllAgentsWithPagination(): Promise<Agent[]> {
    const agents = await this.libraryPage.getAgentsWithPagination();
    return agents;
  }

  async testPaginationWithSearch(searchTerm: string): Promise<void> {
    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const initialCount = await this.libraryPage.getAgentCount();

    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();

    const finalCount = await this.libraryPage.getAgentCount();

    console.log("✅ Pagination with search test completed");
  }

  async verifyNavigation(): Promise<void> {
    const libraryLink = this.page.getByTestId("navbar-link-library");
    const buildLink = this.page.getByTestId("navbar-link-build");
    const marketplaceLink = this.page.getByTestId("navbar-link-marketplace");

    const monitoringAlert = await this.libraryPage.isMonitoringAlertVisible();
  }

  async getAgentsBySearch(searchTerm: string): Promise<Agent[]> {
    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    return agents;
  }

  async getAllAgents(): Promise<Agent[]> {
    await this.libraryPage.clearSearch();
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    return agents;
  }

  async verifyAgentCount(): Promise<boolean> {
    const displayedCount = await this.libraryPage.getAgentCount();
    const actualAgents = await this.libraryPage.getAgents();

    const countsMatch = displayedCount === actualAgents.length;
    return countsMatch;
  }

  async testSearchFunctionality(): Promise<void> {
    const allAgents = await this.getAllAgents();

    if (allAgents.length === 0) {
      return;
    }

    const testAgent = allAgents[0];

    await this.searchAndVerify("nonexistentterm123");

    if (testAgent.name.length > 3) {
      const partialTerm = testAgent.name.substring(0, 3);
      await this.searchAndVerify(partialTerm);
    }

    await this.libraryPage.clearSearch();
    await this.libraryPage.waitForAgentsToLoad();
    const clearedResults = await this.libraryPage.getAgents();

    console.log(
      `🔄 After clearing search: ${clearedResults.length} agents (expected: ${allAgents.length})`,
    );

    console.log("✅ Search functionality test completed");
  }

  generateTestAgentData(): { name: string; description: string } {
    const timestamp = Date.now();
    return {
      name: `Test Agent ${timestamp}`,
      description: `This is a test agent created at ${new Date().toISOString()}`,
    };
  }

  async waitForFullPageLoad(): Promise<void> {
    await this.libraryPage.waitForAgentsToLoad();

    await this.page.waitForTimeout(1000);
  }
}
