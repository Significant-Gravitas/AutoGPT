import { Page } from "@playwright/test";
import { LibraryPage, Agent } from "../pages/library.page";
import { AgentCreationService } from "./agent-creation";

/**
 * Utility functions for library page tests
 */
export class LibraryUtils {
  private agentCreationService: AgentCreationService;

  constructor(
    private page: Page,
    private libraryPage: LibraryPage,
  ) {
    this.agentCreationService = new AgentCreationService(page);
  }

  /**
   * Navigate to library page and wait for it to load
   */
  async navigateToLibrary(): Promise<void> {
    console.log("🔗 Navigating to library page");
    await this.page.goto("/library");
    await this.libraryPage.waitForPageLoad();
    await this.libraryPage.isLoaded();
    console.log("✅ Library page loaded successfully");
  }

  /**
   * Create test agents using builder page
   */
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

  /**
   * Setup library page with test data
   */
  async setupLibraryWithTestData(agentCount: number = 3): Promise<Agent[]> {
    console.log(`📋 Setting up library page with ${agentCount} test agents`);

    // Create test agents first using builder page
    await this.createTestAgents(agentCount);

    // Navigate to library page
    await this.navigateToLibrary();

    // Wait for agents to load
    await this.libraryPage.waitForAgentsToLoad();

    // Get the actual agents from the library (with correct IDs and URLs)
    const libraryAgents = await this.libraryPage.getAgents();

    console.log(
      `✅ Library setup complete with ${libraryAgents.length} agents`,
    );
    return libraryAgents;
  }

  /**
   * Ensure minimum agent count for testing
   */
  async ensureMinimumAgents(minCount: number = 3): Promise<Agent[]> {
    console.log(`🔍 Ensuring minimum ${minCount} agents for testing`);

    // Navigate to library first
    await this.navigateToLibrary();
    await this.libraryPage.waitForAgentsToLoad();

    // Check current agent count
    const currentAgents = await this.libraryPage.getAgents();
    const currentCount = currentAgents.length;

    console.log(`📊 Current agent count: ${currentCount}`);

    if (currentCount < minCount) {
      const needed = minCount - currentCount;
      console.log(`🏗️ Creating ${needed} additional agents`);

      // Create additional agents using AgentCreationService
      await this.agentCreationService.createMultipleAgents(
        needed,
        "Library Test Agent",
        "Agent created to ensure minimum count for testing",
      );

      // Refresh library page
      await this.navigateToLibrary();
      await this.libraryPage.waitForAgentsToLoad();

      // Get updated agent list
      const updatedAgents = await this.libraryPage.getAgents();
      console.log(`✅ Updated agent count: ${updatedAgents.length}`);
      return updatedAgents;
    }

    console.log(`✅ Sufficient agents already exist`);
    return currentAgents;
  }

  /**
   * Create a single test agent
   */
  async createSingleTestAgent(customName?: string): Promise<void> {
    console.log(`🏗️ Creating single test agent`);

    const agentName = customName || `Test Agent ${Date.now()}`;
    const agentDescription = `Test agent created for library testing`;

    // Use AgentCreationService to create the agent
    await this.agentCreationService.createSimpleAgent(
      agentName,
      agentDescription,
    );

    console.log(`✅ Created single test agent: ${agentName}`);
  }

  /**
   * Setup library page ensuring it has data for testing
   */
  async setupLibraryForTesting(): Promise<Agent[]> {
    console.log(`🚀 Setting up library page for comprehensive testing`);

    // First, ensure we have enough agents for testing
    const agents = await this.ensureMinimumAgents(5);

    // If we still don't have enough, create more
    if (agents.length < 3) {
      console.log(`🏗️ Creating additional agents as fallback`);
      await this.agentCreationService.createMultipleAgents(
        3,
        "Fallback Agent",
        "Fallback agent created for testing",
      );

      // Refresh and get updated list
      await this.navigateToLibrary();
      await this.libraryPage.waitForAgentsToLoad();
      return await this.libraryPage.getAgents();
    }

    console.log(`✅ Library setup complete with ${agents.length} agents`);
    return agents;
  }

  /**
   * Perform a search and verify results
   */
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

    // Verify search results contain the search term
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

  /**
   * Test sorting functionality
   */
  async testSorting(): Promise<void> {
    console.log("🔄 Testing sorting functionality");

    // Test Creation Date sorting
    await this.libraryPage.selectSortOption("Creation Date");
    let currentSort = await this.libraryPage.getCurrentSortOption();
    console.log(`📅 Current sort option: ${currentSort}`);

    const agentsCreationDate = await this.libraryPage.getAgents();

    // Test Last Modified sorting
    await this.libraryPage.selectSortOption("Last Modified");
    currentSort = await this.libraryPage.getCurrentSortOption();
    console.log(`📝 Current sort option: ${currentSort}`);

    const agentsLastModified = await this.libraryPage.getAgents();

    // Verify sorting changed the order (if there are multiple agents)
    if (agentsCreationDate.length > 1 && agentsLastModified.length > 1) {
      const orderChanged =
        agentsCreationDate[0].id !== agentsLastModified[0].id;
      console.log(`🔄 Sort order changed: ${orderChanged}`);
    }

    console.log("✅ Sorting test completed");
  }

  /**
   * Test upload dialog functionality
   */
  async testUploadDialog(): Promise<void> {
    console.log("📤 Testing upload dialog");

    // Open upload dialog
    await this.libraryPage.openUploadDialog();
    const isVisible = await this.libraryPage.isUploadDialogVisible();
    console.log(`👁️ Upload dialog visible: ${isVisible}`);

    // Test form filling
    const testName = "Test Agent";
    const testDescription = "This is a test agent description";

    await this.libraryPage.fillUploadForm(testName, testDescription);

    // Check if upload button is enabled (should be disabled without file)
    const isUploadEnabled = await this.libraryPage.isUploadButtonEnabled();
    console.log(`🔘 Upload button enabled: ${isUploadEnabled}`);

    // Close dialog
    await this.libraryPage.closeUploadDialog();
    const isHidden = !(await this.libraryPage.isUploadDialogVisible());
    console.log(`👁️ Upload dialog hidden: ${isHidden}`);

    console.log("✅ Upload dialog test completed");
  }

  /**
   * Test agent interactions
   */
  async testAgentInteractions(): Promise<void> {
    console.log("🤖 Testing agent interactions");

    const agents = await this.libraryPage.getAgents();

    if (agents.length === 0) {
      console.log("⚠️ No agents found to test interactions");
      return;
    }

    const testAgent = agents[0];
    console.log(`🎯 Testing interactions with agent: ${testAgent.name}`);

    // Test agent visibility
    const isVisible = await this.libraryPage.isAgentVisible(testAgent);
    console.log(`👁️ Agent visible: ${isVisible}`);

    // For now, just verify the agent card elements are present
    const agentHeading = this.page.getByRole("heading", {
      name: testAgent.name,
      level: 3,
    });
    const isHeadingVisible = await agentHeading.isVisible();
    console.log(`📝 Agent heading visible: ${isHeadingVisible}`);

    console.log("✅ Agent interactions test completed");
  }

  /**
   * Test pagination functionality
   */
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

  /**
   * Test infinite scroll functionality
   */
  async testInfiniteScroll(): Promise<void> {
    console.log("🔄 Testing infinite scroll functionality");

    const initialCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Initial agent count: ${initialCount}`);

    // Test scrolling to load more
    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();
    console.log(`📊 New agents loaded: ${newAgentsLoaded}`);

    const finalCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Final agent count: ${finalCount}`);

    // Test if pagination is working
    const isPaginationWorking = await this.libraryPage.isPaginationWorking();
    console.log(`📄 Pagination working: ${isPaginationWorking}`);

    console.log("✅ Infinite scroll test completed");
  }

  /**
   * Get all agents with pagination
   */
  async getAllAgentsWithPagination(): Promise<Agent[]> {
    console.log("📋 Getting all agents with pagination");

    const agents = await this.libraryPage.getAgentsWithPagination();
    console.log(`📊 Total agents found: ${agents.length}`);

    return agents;
  }

  /**
   * Test pagination with search
   */
  async testPaginationWithSearch(searchTerm: string): Promise<void> {
    console.log(`🔍 Testing pagination with search: "${searchTerm}"`);

    // First search
    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const initialCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Initial search results: ${initialCount}`);

    // Test pagination with search results
    const newAgentsLoaded = await this.libraryPage.scrollAndWaitForNewAgents();
    console.log(`📊 New search results loaded: ${newAgentsLoaded}`);

    const finalCount = await this.libraryPage.getAgentCount();
    console.log(`📊 Final search results: ${finalCount}`);

    console.log("✅ Pagination with search test completed");
  }

  /**
   * Verify page navigation elements
   */
  async verifyNavigation(): Promise<void> {
    console.log("🧭 Verifying navigation elements");

    // Check if navbar links are present
    const libraryLink = this.page.getByTestId("navbar-link-library");
    const buildLink = this.page.getByTestId("navbar-link-build");
    const marketplaceLink = this.page.getByTestId("navbar-link-marketplace");

    console.log(`🔗 Library link present: ${await libraryLink.isVisible()}`);
    console.log(`🔗 Build link present: ${await buildLink.isVisible()}`);
    console.log(
      `🔗 Marketplace link present: ${await marketplaceLink.isVisible()}`,
    );

    // Check monitoring alert
    const monitoringAlert = await this.libraryPage.isMonitoringAlertVisible();
    console.log(`⚠️ Monitoring alert visible: ${monitoringAlert}`);

    console.log("✅ Navigation verification completed");
  }

  /**
   * Get agents by search term
   */
  async getAgentsBySearch(searchTerm: string): Promise<Agent[]> {
    console.log(`🔍 Getting agents by search term: "${searchTerm}"`);

    await this.libraryPage.searchAgents(searchTerm);
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    console.log(`📊 Found ${agents.length} agents matching "${searchTerm}"`);

    return agents;
  }

  /**
   * Clear search and get all agents
   */
  async getAllAgents(): Promise<Agent[]> {
    console.log("📋 Getting all agents");

    await this.libraryPage.clearSearch();
    await this.libraryPage.waitForAgentsToLoad();

    const agents = await this.libraryPage.getAgents();
    console.log(`📊 Found ${agents.length} total agents`);

    return agents;
  }

  /**
   * Verify agent count matches displayed count
   */
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

  /**
   * Test search functionality thoroughly
   */
  async testSearchFunctionality(): Promise<void> {
    console.log("🔍 Testing search functionality comprehensively");

    // Get all agents first
    const allAgents = await this.getAllAgents();
    console.log(`📊 Total agents available: ${allAgents.length}`);

    if (allAgents.length === 0) {
      console.log("⚠️ No agents available for search testing");
      return;
    }

    // Test searching with existing agent name
    const testAgent = allAgents[0];

    // Test searching with non-existent term
    await this.searchAndVerify("nonexistentterm123");

    // Test partial search
    if (testAgent.name.length > 3) {
      const partialTerm = testAgent.name.substring(0, 3);
      await this.searchAndVerify(partialTerm);
    }

    // Clear search and verify all agents return
    await this.libraryPage.clearSearch();
    await this.libraryPage.waitForAgentsToLoad();
    const clearedResults = await this.libraryPage.getAgents();

    console.log(
      `🔄 After clearing search: ${clearedResults.length} agents (expected: ${allAgents.length})`,
    );

    console.log("✅ Search functionality test completed");
  }

  /**
   * Generate test data for uploads
   */
  generateTestAgentData(): { name: string; description: string } {
    const timestamp = Date.now();
    return {
      name: `Test Agent ${timestamp}`,
      description: `This is a test agent created at ${new Date().toISOString()}`,
    };
  }

  /**
   * Wait for page to be fully loaded with agents
   */
  async waitForFullPageLoad(): Promise<void> {
    console.log("⏳ Waiting for full page load");

    await this.libraryPage.waitForPageLoad();
    await this.libraryPage.waitForAgentsToLoad();

    // Wait a bit more for any animations or loading states
    await this.page.waitForTimeout(1000);

    console.log("✅ Full page load completed");
  }
}

/**
 * Helper function to create LibraryUtils instance
 */
export function createLibraryUtils(page: Page): LibraryUtils {
  const libraryPage = new LibraryPage(page);
  return new LibraryUtils(page, libraryPage);
}

/**
 * Agent creation utilities for CI/testing environments
 */
// Legacy AgentCreationUtils class - use AgentCreationService instead
export class AgentCreationUtils {
  private agentCreationService: AgentCreationService;

  constructor(private page: Page) {
    this.agentCreationService = new AgentCreationService(page);
  }

  /**
   * Create multiple test agents quickly
   * @deprecated Use AgentCreationService.createMultipleAgents instead
   */
  async createMultipleAgents(count: number): Promise<string[]> {
    console.log(`🏭 Creating ${count} agents for testing (legacy method)`);
    return await this.agentCreationService.createMultipleAgents(
      count,
      "CI-Test-Agent",
      "CI test agent for library testing",
    );
  }

  /**
   * Create agents with specific names for testing
   * @deprecated Use AgentCreationService.createNamedAgents instead
   */
  async createNamedAgents(names: string[]): Promise<void> {
    console.log(`📝 Creating ${names.length} named agents (legacy method)`);
    await this.agentCreationService.createNamedAgents(names, "Test agent");
  }
}

/**
 * Agent search utilities
 */
export class AgentSearchUtils {
  /**
   * Filter agents by name
   */
  static filterByName(agents: Agent[], searchTerm: string): Agent[] {
    return agents.filter((agent) =>
      agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }

  /**
   * Filter agents by description
   */
  static filterByDescription(agents: Agent[], searchTerm: string): Agent[] {
    return agents.filter((agent) =>
      agent.description.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }

  /**
   * Filter agents by either name or description
   */
  static filterByNameOrDescription(
    agents: Agent[],
    searchTerm: string,
  ): Agent[] {
    return agents.filter(
      (agent) =>
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }

  /**
   * Sort agents by name
   */
  static sortByName(agents: Agent[], ascending: boolean = true): Agent[] {
    return [...agents].sort((a, b) => {
      const comparison = a.name.localeCompare(b.name);
      return ascending ? comparison : -comparison;
    });
  }

  /**
   * Get unique agent names
   */
  static getUniqueNames(agents: Agent[]): string[] {
    return [...new Set(agents.map((agent) => agent.name))];
  }

  /**
   * Validate agent data
   */
  static validateAgent(agent: Agent): boolean {
    return !!(
      agent.id &&
      agent.name &&
      agent.description &&
      agent.seeRunsUrl &&
      agent.openInBuilderUrl
    );
  }
}
