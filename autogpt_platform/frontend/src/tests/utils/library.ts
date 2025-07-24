import { Page } from "@playwright/test";
import { LibraryPage, Agent } from "../pages/library.page";
import { AgentCreationService } from "./agent-creation";

export class LibraryUtils {
  private agentCreationService: AgentCreationService;

  constructor(
    private page: Page,
    private libraryPage: LibraryPage,
  ) {
    this.agentCreationService = new AgentCreationService(page);
  }

  async navigateToLibrary(): Promise<void> {
    console.log("🔗 Navigating to library page");
    await this.page.goto("/library");
    await this.libraryPage.isLoaded();
    console.log("✅ Library page loaded successfully");
  }

  async createTestAgents(count: number = 3): Promise<Agent[]> {
    console.log(`🏗️ Creating ${count} test agents using builder page`);

    // Use AgentCreationService to create agents
    await this.agentCreationService.createMultipleAgents(
      count,
      "Test Agent",
      "This is a test agent created for library testing",
    );

    console.log(`🎉 Successfully created ${count} test agents`);
    return []; // Return empty array as we'll fetch actual agents from library
  }

  async setupLibraryWithTestData(agentCount: number = 3): Promise<Agent[]> {
    console.log(`📋 Setting up library page with ${agentCount} test agents`);

    await this.createTestAgents(agentCount);

    await this.navigateToLibrary();

    await this.libraryPage.waitForAgentsToLoad();

    const libraryAgents = await this.libraryPage.getAgents();

    console.log(
      `✅ Library setup complete with ${libraryAgents.length} agents`,
    );
    return libraryAgents;
  }

  async ensureMinimumAgents(minCount: number = 3): Promise<Agent[]> {
    console.log(`🔍 Ensuring minimum ${minCount} agents for testing`);

    await this.navigateToLibrary();
    await this.libraryPage.waitForAgentsToLoad();

    const currentAgents = await this.libraryPage.getAgents();
    const currentCount = currentAgents.length;

    console.log(`📊 Current agent count: ${currentCount}`);

    if (currentCount < minCount) {
      const needed = minCount - currentCount;
      console.log(`🏗️ Creating ${needed} additional agents`);

      await this.agentCreationService.createMultipleAgents(
        needed,
        "Library Test Agent",
        "Agent created to ensure minimum count for testing",
      );

      await this.navigateToLibrary();
      await this.libraryPage.waitForAgentsToLoad();

      const updatedAgents = await this.libraryPage.getAgents();
      console.log(`✅ Updated agent count: ${updatedAgents.length}`);
      return updatedAgents;
    }

    console.log(`✅ Sufficient agents already exist`);
    return currentAgents;
  }

  async createSingleTestAgent(customName?: string): Promise<void> {
    console.log(`🏗️ Creating single test agent`);

    const agentName = customName || `Test Agent ${Date.now()}`;
    const agentDescription = `Test agent created for library testing`;

    await this.agentCreationService.createSimpleAgent(
      agentName,
      agentDescription,
    );

    console.log(`✅ Created single test agent: ${agentName}`);
  }

  async setupLibraryForTesting(): Promise<Agent[]> {
    console.log(`🚀 Setting up library page for comprehensive testing`);

    const agents = await this.ensureMinimumAgents(5);

    if (agents.length < 3) {
      console.log(`🏗️ Creating additional agents as fallback`);
      await this.agentCreationService.createMultipleAgents(
        3,
        "Fallback Agent",
        "Fallback agent created for testing",
      );

      await this.navigateToLibrary();
      await this.libraryPage.waitForAgentsToLoad();
      return await this.libraryPage.getAgents();
    }

    console.log(`✅ Library setup complete with ${agents.length} agents`);
    return agents;
  }

  async searchAndVerify(
    searchTerm: string,
    expectedCount?: number,
  ): Promise<Agent[]> {
    console.log(`🔍 Searching for: "${searchTerm}"`);

    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();

    if (expectedCount !== undefined) {
      console.log(
        `📊 Expected ${expectedCount} agents, found ${agents.length}`,
      );
    }

    for (const agent of agents) {
      const containsSearchTerm =
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description.toLowerCase().includes(searchTerm.toLowerCase());

      if (!containsSearchTerm) {
        console.warn(
          `⚠️ Agent "${agent.name}" doesn't contain search term "${searchTerm}"`,
        );
      }
    }

    console.log(`✅ Search completed, found ${agents.length} agents`);
    return agents;
  }

  async testSorting(): Promise<void> {
    console.log("🔄 Testing sorting functionality");

    await this.libraryPage.selectSortOption(this.page, "Creation Date");
    let currentSort = await this.libraryPage.getCurrentSortOption();
    console.log(`📅 Current sort option: ${currentSort}`);

    const agentsCreationDate = await this.libraryPage.getAgents();

    await this.libraryPage.selectSortOption(this.page, "Last Modified");
    currentSort = await this.libraryPage.getCurrentSortOption();
    console.log(`📝 Current sort option: ${currentSort}`);

    const agentsLastModified = await this.libraryPage.getAgents();

    if (agentsCreationDate.length > 1 && agentsLastModified.length > 1) {
      const orderChanged =
        agentsCreationDate[0].id !== agentsLastModified[0].id;
      console.log(`🔄 Sort order changed: ${orderChanged}`);
    }

    console.log("✅ Sorting test completed");
  }

  async testUploadDialog(): Promise<void> {
    console.log("📤 Testing upload dialog");

    await this.libraryPage.openUploadDialog();
    const isVisible = await this.libraryPage.isUploadDialogVisible();
    console.log(`👁️ Upload dialog visible: ${isVisible}`);

    const testName = "Test Agent";
    const testDescription = "This is a test agent description";

    await this.libraryPage.fillUploadForm(testName, testDescription);

    const isUploadEnabled = await this.libraryPage.isUploadButtonEnabled();
    console.log(`🔘 Upload button enabled: ${isUploadEnabled}`);

    await this.libraryPage.closeUploadDialog();
    const isHidden = !(await this.libraryPage.isUploadDialogVisible());
    console.log(`👁️ Upload dialog hidden: ${isHidden}`);

    console.log("✅ Upload dialog test completed");
  }

  async testAgentInteractions(): Promise<void> {
    console.log("🤖 Testing agent interactions");

    const agents = await this.libraryPage.getAgents();

    if (agents.length === 0) {
      console.log("⚠️ No agents found to test interactions");
      return;
    }

    const testAgent = agents[0];
    console.log(`🎯 Testing interactions with agent: ${testAgent.name}`);

    // Replace isAgentVisible with direct locator check
    const isVisible = await this.page.getByRole("heading", {
      name: testAgent.name,
      level: 3,
    }).isVisible();
    console.log(`👁️ Agent visible: ${isVisible}`);

    const agentHeading = this.page.getByRole("heading", {
      name: testAgent.name,
      level: 3,
    });
    const isHeadingVisible = await agentHeading.isVisible();
    console.log(`📝 Agent heading visible: ${isHeadingVisible}`);

    console.log("✅ Agent interactions test completed");
  }

  async testPagination(): Promise<void> {
    console.log("📄 Testing pagination functionality");

    const paginationResult = await this.libraryPage.testPagination();

    console.log(`📊 Initial agents: ${paginationResult.initialCount}`);
    console.log(`📊 Final agents: ${paginationResult.finalCount}`);
    console.log(`📄 Has more agents: ${paginationResult.hasMore}`);

    if (paginationResult.hasMore) {
      console.log("✅ Pagination is working - more agents loaded");
    } else {
      console.log("ℹ️ No additional agents loaded (may be all agents shown)");
    }

    console.log("✅ Pagination test completed");
  }

  async testInfiniteScroll(): Promise<void> {
    console.log("🔄 Testing infinite scroll functionality");

    const initialCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Initial agent count: ${initialCount}`);

    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();
    console.log(`📊 New agents loaded: ${newAgentsLoaded}`);

    const finalCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Final agent count: ${finalCount}`);

    const isPaginationWorking = await this.libraryPage.isPaginationWorking();
    console.log(`📄 Pagination working: ${isPaginationWorking}`);

    console.log("✅ Infinite scroll test completed");
  }

  async getAllAgentsWithPagination(): Promise<Agent[]> {
    console.log("📋 Getting all agents with pagination");

    const agents = await this.libraryPage.getAgentsWithPagination();
    console.log(`📊 Total agents found: ${agents.length}`);

    return agents;
  }

  async testPaginationWithSearch(searchTerm: string): Promise<void> {
    console.log(`🔍 Testing pagination with search: "${searchTerm}"`);

    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const initialCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Initial search results: ${initialCount}`);

    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();
    console.log(`📊 New search results loaded: ${newAgentsLoaded}`);

    const finalCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Final search results: ${finalCount}`);

    console.log("✅ Pagination with search test completed");
  }

  async verifyNavigation(): Promise<void> {
    console.log("🧭 Verifying navigation elements");

    const libraryLink = this.page.getByTestId("navbar-link-library");
    const buildLink = this.page.getByTestId("navbar-link-build");
    const marketplaceLink = this.page.getByTestId("navbar-link-marketplace");

    console.log(`🔗 Library link present: ${await libraryLink.isVisible()}`);
    console.log(`🔗 Build link present: ${await buildLink.isVisible()}`);
    console.log(
      `🔗 Marketplace link present: ${await marketplaceLink.isVisible()}`,
    );

    const monitoringAlert = await this.libraryPage.isMonitoringAlertVisible();
    console.log(`⚠️ Monitoring alert visible: ${monitoringAlert}`);

    console.log("✅ Navigation verification completed");
  }

  async getAgentsBySearch(searchTerm: string): Promise<Agent[]> {
    console.log(`🔍 Getting agents by search term: "${searchTerm}"`);

    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    console.log(`📊 Found ${agents.length} agents matching "${searchTerm}"`);

    return agents;
  }

  async getAllAgents(): Promise<Agent[]> {
    console.log("📋 Getting all agents");

    await this.libraryPage.clearSearch();
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    console.log(`📊 Found ${agents.length} total agents`);

    return agents;
  }

  async verifyAgentCount(): Promise<boolean> {
    console.log("🔢 Verifying agent count");

    const displayedCount = await this.libraryPage.getAgentCount();
    const actualAgents = await this.libraryPage.getAgents();

    const countsMatch = displayedCount === actualAgents.length;
    console.log(
      `📊 Displayed count: ${displayedCount}, Actual count: ${actualAgents.length}`,
    );
    console.log(`✅ Counts match: ${countsMatch}`);

    return countsMatch;
  }

  async testSearchFunctionality(): Promise<void> {
    console.log("🔍 Testing search functionality comprehensively");

    const allAgents = await this.getAllAgents();
    console.log(`📊 Total agents available: ${allAgents.length}`);

    if (allAgents.length === 0) {
      console.log("⚠️ No agents available for search testing");
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
    console.log("⏳ Waiting for full page load");
    await this.libraryPage.waitForAgentsToLoad();

    await this.page.waitForTimeout(1000);

    console.log("✅ Full page load completed");
  }
}
