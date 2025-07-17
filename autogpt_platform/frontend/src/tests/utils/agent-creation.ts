import { Page } from "@playwright/test";
import { BuildPage, Block } from "../pages/build.page";
import { LibraryPage } from "../pages/library.page";
import { LoginPage } from "../pages/login.page";
import { LibraryUtils } from "./library";
import { loadUserPool } from "./auth";
import { getBrowser } from "./get-browser";
import fs from "fs";
import path from "path";

/**
 * Comprehensive agent creation utility using BuildPage
 * Handles all aspects of creating test agents through the builder interface
 */
export class AgentCreationService {
  private buildPage: BuildPage;

  constructor(private page: Page) {
    this.buildPage = new BuildPage(page);
  }

  /**
   * Create a single agent with a simple block
   */
  async createSimpleAgent(
    name: string,
    description: string = "",
    haveTutorial: boolean = true,
    blockType: "dictionary" | "calculator" | "custom" = "dictionary",
  ): Promise<void> {
    console.log(`🏗️ Creating agent: ${name}`);

    try {
      // Navigate to build page
      await this.buildPage.navbar.clickBuildLink();
      await this.page.waitForURL(/^.*\/build$/);

      if (haveTutorial) {
        await this.buildPage.closeTutorial();
      }

      // Open blocks panel
      await this.buildPage.openBlocksPanel();

      // Get the appropriate block
      let block: Block;
      switch (blockType) {
        case "calculator":
          block = await this.buildPage.getCalculatorBlockDetails();
          break;
        case "dictionary":
        default:
          block = await this.buildPage.getDictionaryBlockDetails();
          break;
      }

      // Add the block to the canvas
      await this.buildPage.addBlock(block);

      // Close blocks panel
      await this.buildPage.closeBlocksPanel();

      // Save the agent
      await this.buildPage.saveAgent(name, description);

      // Wait for save to complete
      await this.buildPage.waitForVersionField();

      await this.buildPage.navbar.clickMarketplaceLink();

      console.log(`✅ Created agent: ${name}`);
    } catch (error) {
      console.error(`❌ Failed to create simple agent: ${name}`, error);
      throw error;
    }
  }

  /**
   * Create multiple agents efficiently
   */
  async createMultipleAgents(
    count: number,
    namePrefix: string = "Test Agent",
    description: string = "Test agent for library testing",
  ): Promise<string[]> {
    console.log(`🏭 Creating ${count} agents...`);

    const createdAgents: string[] = [];
    const startTime = Date.now();

    try {
      // Create first agent with tutorial
      const firstAgentName = `${namePrefix} ${Date.now()}-${0}`;
      const firstAgentDescription = `${description} (${0 + 1}/${count})`;

      await this.createSimpleAgent(firstAgentName, firstAgentDescription, true);
      createdAgents.push(firstAgentName);

      // Create remaining agents without tutorial
      for (let i = 1; i < count; i++) {
        const agentName = `${namePrefix} ${Date.now()}-${i}`;
        const agentDescription = `${description} (${i + 1}/${count})`;

        await this.createSimpleAgent(agentName, agentDescription, false);
        createdAgents.push(agentName);

        // Wait between creations to avoid overwhelming the system
        await this.page.waitForTimeout(1000);
      }

      const totalDuration = Date.now() - startTime;
      console.log(
        `✅ Created ${createdAgents.length} agents in ${(totalDuration / 1000).toFixed(2)}s`,
      );

      return createdAgents;
    } catch (error) {
      console.error(
        `❌ Failed batch creation after ${createdAgents.length} agents:`,
        error,
      );
      throw error;
    }
  }

  /**
   * Create agents with specific names, need for searching purpose
   */
  async createNamedAgents(
    names: string[],
    descriptionTemplate: string = "Test agent",
  ): Promise<void> {
    console.log(`📝 Creating ${names.length} named agents...`);

    const startTime = Date.now();

    try {
      // Create first agent with tutorial
      await this.createSimpleAgent(
        names[0],
        `${descriptionTemplate}: ${names[0]}`,
        true,
      );

      // Create remaining agents without tutorial
      for (let i = 1; i < names.length; i++) {
        const name = names[i];
        const description = `${descriptionTemplate}: ${name}`;

        await this.createSimpleAgent(name, description, false);

        // Wait between creations to avoid overwhelming the system
        await this.page.waitForTimeout(1000);
      }

      const totalDuration = Date.now() - startTime;
      console.log(
        `✅ Created ${names.length} named agents in ${(totalDuration / 1000).toFixed(2)}s`,
      );
    } catch (error) {
      console.error(`❌ Failed during named agent creation:`, error);
      throw error;
    }
  }

  /**
   * Create agents for specific test scenarios
   */
  async createTestScenarioAgents(): Promise<{
    searchAgents: string[];
    paginationAgents: string[];
    sortAgents: string[];
  }> {
    console.log(`🎯 Creating agents for specific test scenarios`);

    const searchAgents = await this.createMultipleAgents(
      3,
      "Search Test Agent",
      "Agent for testing search functionality",
    );

    const paginationAgents = await this.createMultipleAgents(
      5,
      "Pagination Test Agent",
      "Agent for testing pagination functionality",
    );

    const sortAgents = await this.createMultipleAgents(
      3,
      "Sort Test Agent",
      "Agent for testing sort functionality",
    );

    console.log(`✅ Created all scenario test agents`);

    return {
      searchAgents,
      paginationAgents,
      sortAgents,
    };
  }

  /**
   * Create agents in batches to avoid overwhelming the system
   */
  async createAgentsInBatches(
    totalCount: number,
    batchSize: number = 3,
    batchDelay: number = 2000,
  ): Promise<string[]> {
    console.log(`📦 Creating ${totalCount} agents in batches of ${batchSize}`);

    const allAgents: string[] = [];
    const batches = Math.ceil(totalCount / batchSize);

    for (let batch = 0; batch < batches; batch++) {
      const startIndex = batch * batchSize;
      const endIndex = Math.min(startIndex + batchSize, totalCount);
      const batchCount = endIndex - startIndex;

      const batchAgents = await this.createMultipleAgents(
        batchCount,
        `Batch-${batch + 1}-Agent`,
        `Agent from batch ${batch + 1}`,
      );

      allAgents.push(...batchAgents);

      // Wait between batches if not the last batch
      if (batch < batches - 1) {
        await this.page.waitForTimeout(batchDelay);
      }
    }

    console.log(`✅ Created ${allAgents.length} agents in ${batches} batches`);
    return allAgents;
  }
}

export interface AgentPool {
  agents: string[];
  createdAt: string;
  version: string;
}

/**
 * Create test agents globally for library tests
 * @returns Promise<string[]> - Array of created agent names
 */
export async function createGlobalTestAgents(): Promise<string[]> {
  console.log(`🚀 Starting global test agent creation`);

  const overallStartTime = Date.now();

  try {
    // Initialize browser
    const browser = await getBrowser();
    const context = await browser.newContext({
      baseURL: "http://localhost:3000/",
    });
    const page = await context.newPage();

    try {
      // Initialize services
      const libraryPage = new LibraryPage(page);
      const libraryUtils = new LibraryUtils(page, libraryPage);
      const agentCreationService = new AgentCreationService(page);

      // Login for setup
      await page.goto("/login");
      const loginPage = new LoginPage(page);
      const users_pool = await loadUserPool();

      if (!users_pool) {
        console.error("❌ No user pool found - users must be created first");
        throw new Error("No user pool available for agent creation");
      }

      // Use a test user for agent creation
      const testUser = users_pool.users[0];
      await loginPage.login(testUser.email, testUser.password);
      await page.waitForURL("/marketplace");

      // Create diverse agents for comprehensive testing
      const agentNames = [
        "Email Marketing Agent",
        "Data Analysis Agent",
        "Content Creator Agent",
        "Social Media Manager Agent",
        "Report Generator Agent",
        "Task Automation Agent",
        "Customer Service Bot Agent",
        "Analytics Dashboard Agent",
        "Newsletter Manager Agent",
        "Performance Monitor Agent",
        "Search Optimizer Agent",
        "Workflow Coordinator Agent",
      ];

      await agentCreationService.createNamedAgents(
        agentNames,
        "Pre-created agent for library testing",
      );

      // Navigate to library to ensure agents are available
      await libraryUtils.navigateToLibrary();
      await libraryPage.waitForAgentsToLoad();

      const overallDuration = Date.now() - overallStartTime;
      console.log(
        `✅ Created ${agentNames.length} test agents in ${(overallDuration / 1000).toFixed(2)}s`,
      );

      return agentNames;
    } catch (error) {
      console.error("❌ Error during agent creation process:", error);
      throw error;
    } finally {
      await context.close();
      await browser.close();
    }
  } catch (error) {
    console.error("❌ Failed to create test agents:", error);
    throw error;
  }
}

/**
 * Save agent pool to file system
 * @param agents - Array of agent names to save
 * @param filePath - Path to save the file (optional)
 */
export async function saveAgentPool(
  agents: string[],
  filePath?: string,
): Promise<void> {
  const defaultPath = path.resolve(process.cwd(), ".auth", "agent-pool.json");
  const finalPath = filePath || defaultPath;

  // Ensure .auth directory exists
  const dirPath = path.dirname(finalPath);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  const agentPool: AgentPool = {
    agents,
    createdAt: new Date().toISOString(),
    version: "1.0.0",
  };

  try {
    fs.writeFileSync(finalPath, JSON.stringify(agentPool, null, 2));
    console.log(`✅ Saved agent pool: ${agents.length} agents`);
  } catch (error) {
    console.error(`❌ Failed to save agent pool:`, error);
    throw error;
  }
}

/**
 * Load agent pool from file system
 * @param filePath - Path to load from (optional)
 * @returns Promise<AgentPool | null> - Loaded agent pool or null if not found
 */
export async function loadAgentPool(
  filePath?: string,
): Promise<AgentPool | null> {
  const defaultPath = path.resolve(process.cwd(), ".auth", "agent-pool.json");
  const finalPath = filePath || defaultPath;

  try {
    if (!fs.existsSync(finalPath)) {
      console.log(`⚠️ Agent pool file not found`);
      return null;
    }

    const fileContent = fs.readFileSync(finalPath, "utf-8");
    const agentPool: AgentPool = JSON.parse(fileContent);

    console.log(`✅ Loaded ${agentPool.agents.length} agents from pool`);

    return agentPool;
  } catch (error) {
    console.error(`❌ Failed to load agent pool:`, error);
    return null;
  }
}

/**
 * Creates and saves test agents for library tests
 */
export async function createAndSaveTestAgents() {
  console.log(`🔍 Checking for existing agent pool...`);
  const checkStartTime = Date.now();

  try {
    const existingAgentPool = await loadAgentPool();

    if (existingAgentPool && existingAgentPool.agents.length > 0) {
      console.log(
        `♻️ Using existing agent pool: ${existingAgentPool.agents.length} agents`,
      );
      return;
    }

    console.log(`📋 No existing agent pool found - creating new agents`);

    // Create test agents
    const agents = await createGlobalTestAgents();

    if (agents.length === 0) {
      throw new Error("Failed to create any test agents");
    }

    // Save agent pool
    await saveAgentPool(agents);

    const totalDuration = Date.now() - checkStartTime;
    console.log(
      `✅ Agent creation completed: ${agents.length} agents in ${(totalDuration / 1000).toFixed(2)}s`,
    );
  } catch (error) {
    console.error("❌ Failed to create and save test agents:", error);
    throw error;
  }
}
