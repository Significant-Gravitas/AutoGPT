import { expect, Locator, Page } from "@playwright/test";
import { getSeededTestUser } from "../credentials/accounts";
import { getSelectors } from "../utils/selectors";
import { BasePage } from "./base.page";

export interface Agent {
  id: string;
  name: string;
  description: string;
  imageUrl?: string;
  seeRunsUrl: string;
  openInBuilderUrl: string;
}

export class LibraryPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async isLoaded(): Promise<boolean> {
    console.log(`checking if library page is loaded`);
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });

      await this.page.waitForSelector('[data-testid="library-textbox"]', {
        state: "visible",
        timeout: 10_000,
      });

      console.log("Library page is loaded successfully");
      return true;
    } catch (error) {
      console.log("Library page failed to load:", error);
      return false;
    }
  }

  async navigateToLibrary(): Promise<void> {
    await this.page.goto("/library");
    await this.isLoaded();
  }

  async getAgentCount(): Promise<number> {
    const { getId } = getSelectors(this.page);
    const countText = await getId("agents-count").textContent();
    const match = countText?.match(/^(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  }

  async getAgentCountByListLength(): Promise<number> {
    const { getId } = getSelectors(this.page);
    const agentCards = await getId("library-agent-card").all();
    return agentCards.length;
  }

  async searchAgents(searchTerm: string): Promise<void> {
    console.log(`searching for agents with term: ${searchTerm}`);
    const { getRole } = getSelectors(this.page);
    const searchInput = getRole("textbox", "Search agents");
    await searchInput.fill(searchTerm);

    await this.page.waitForTimeout(500);
  }

  async clearSearch(): Promise<void> {
    console.log(`clearing search`);
    try {
      // Look for the clear button (X icon)
      const clearButton = this.page.locator(".lucide.lucide-x");
      if (await clearButton.isVisible()) {
        await clearButton.click();
      } else {
        // If no clear button, clear the search input directly
        const searchInput = this.page.getByRole("textbox", {
          name: "Search agents",
        });
        await searchInput.fill("");
      }

      // Wait for results to update
      await this.page.waitForTimeout(500);
    } catch (error) {
      console.error("Error clearing search:", error);
    }
  }

  async selectSortOption(
    page: Page,
    sortOption: "Creation Date" | "Last Modified",
  ): Promise<void> {
    const { getRole } = getSelectors(page);
    await getRole("combobox").click();

    await getRole("option", sortOption).click();

    await this.page.waitForTimeout(500);
  }

  async getCurrentSortOption(): Promise<string> {
    console.log(`getting current sort option`);
    try {
      const sortCombobox = this.page.getByRole("combobox");
      const currentOption = await sortCombobox.textContent();
      return currentOption?.trim() || "";
    } catch (error) {
      console.error("Error getting current sort option:", error);
      return "";
    }
  }

  async openUploadDialog(): Promise<void> {
    console.log(`opening upload dialog`);
    // Open the unified Import dialog first
    await this.page.getByRole("button", { name: "Import" }).click();

    // Wait for dialog to appear
    await this.page.getByRole("dialog", { name: "Import" }).waitFor({
      state: "visible",
      timeout: 5_000,
    });

    // Click the "AutoGPT agent" tab
    await this.page.getByRole("tab", { name: "AutoGPT agent" }).click();
  }

  async closeUploadDialog(): Promise<void> {
    await this.page.getByRole("button", { name: "Close" }).click();

    await this.page.getByRole("dialog", { name: "Import" }).waitFor({
      state: "hidden",
      timeout: 5_000,
    });
  }

  async isUploadDialogVisible(): Promise<boolean> {
    console.log(`checking if upload dialog is visible`);
    try {
      const dialog = this.page.getByRole("dialog", { name: "Import" });
      return await dialog.isVisible();
    } catch {
      return false;
    }
  }

  async fillUploadForm(agentName: string, description: string): Promise<void> {
    console.log(
      `filling upload form with name: ${agentName}, description: ${description}`,
    );

    // Fill agent name
    await this.page
      .getByRole("textbox", { name: "Agent name" })
      .fill(agentName);

    // Fill description
    await this.page
      .getByRole("textbox", { name: "Agent description" })
      .fill(description);
  }

  async isUploadButtonEnabled(): Promise<boolean> {
    console.log(`checking if upload button is enabled`);
    try {
      const uploadButton = this.page.getByRole("button", {
        name: "Upload",
      });
      return await uploadButton.isEnabled();
    } catch {
      return false;
    }
  }

  async getAgents(): Promise<Agent[]> {
    const { getId } = getSelectors(this.page);
    const agents: Agent[] = [];

    await getId("library-agent-card")
      .first()
      .waitFor({ state: "visible", timeout: 10_000 });
    const agentCards = await getId("library-agent-card").all();

    for (const card of agentCards) {
      const name = await getId("library-agent-card-name", card).textContent();
      const seeRunsLink = getId("library-agent-card-see-runs-link", card);
      const openInBuilderLink = getId(
        "library-agent-card-open-in-builder-link",
        card,
      );

      const seeRunsUrl = await seeRunsLink.getAttribute("href");

      // Check if the "Open in builder" link exists before getting its href
      const openInBuilderLinkCount = await openInBuilderLink.count();
      const openInBuilderUrl =
        openInBuilderLinkCount > 0
          ? await openInBuilderLink.getAttribute("href")
          : null;

      if (name && seeRunsUrl) {
        const idMatch = seeRunsUrl.match(/\/library\/agents\/([^\/]+)/);
        const id = idMatch ? idMatch[1] : "";

        agents.push({
          id,
          name: name.trim(),
          description: "", // Description is not currently rendered in the card
          seeRunsUrl,
          openInBuilderUrl: openInBuilderUrl || "",
        });
      }
    }

    console.log(`found ${agents.length} agents`);
    return agents;
  }

  async clickAgent(agent: Agent): Promise<void> {
    const { getId } = getSelectors(this.page);
    const nameElement = getId("library-agent-card-name").filter({
      hasText: agent.name,
    });
    await nameElement.first().click();
  }

  async clickSeeRuns(agent: Agent): Promise<void> {
    console.log(`clicking see runs for agent: ${agent.name}`);

    const { getId } = getSelectors(this.page);
    const agentCard = getId("library-agent-card").filter({
      hasText: agent.name,
    });
    const seeRunsLink = getId("library-agent-card-see-runs-link", agentCard);
    await seeRunsLink.first().click();
  }

  async clickOpenInBuilder(agent: Agent): Promise<void> {
    console.log(`clicking open in builder for agent: ${agent.name}`);

    const { getId } = getSelectors(this.page);
    const agentCard = getId("library-agent-card").filter({
      hasText: agent.name,
    });
    const builderLink = getId(
      "library-agent-card-open-in-builder-link",
      agentCard,
    );
    await builderLink.first().click();
  }

  async waitForAgentsToLoad(): Promise<void> {
    const { getId } = getSelectors(this.page);
    await Promise.race([
      getId("library-agent-card")
        .first()
        .waitFor({ state: "visible", timeout: 10_000 }),
      getId("agents-count").waitFor({ state: "visible", timeout: 10_000 }),
    ]);
  }

  async getSearchValue(): Promise<string> {
    console.log(`getting search input value`);
    try {
      const searchInput = this.page.getByRole("textbox", {
        name: "Search agents",
      });
      return await searchInput.inputValue();
    } catch {
      return "";
    }
  }

  async hasNoAgentsMessage(): Promise<boolean> {
    const { getText } = getSelectors(this.page);
    const noAgentsText = getText("0 agents");
    return noAgentsText !== null;
  }

  async scrollToBottom(): Promise<void> {
    console.log(`scrolling to bottom to trigger pagination`);
    await this.page.keyboard.press("End");
    await this.page.waitForTimeout(1000);
  }

  async scrollDown(): Promise<void> {
    console.log(`scrolling down to trigger pagination`);
    await this.page.keyboard.press("PageDown");
    await this.page.waitForTimeout(1000);
  }

  async scrollToLoadMore(): Promise<void> {
    console.log(`scrolling to load more agents`);

    const initialCount = await this.getAgentCountByListLength();
    console.log(`Initial agent count (DOM cards): ${initialCount}`);

    await this.scrollToBottom();

    await this.page
      .waitForLoadState("networkidle", { timeout: 10000 })
      .catch(() => console.log("Network idle timeout, continuing..."));

    await this.page
      .waitForFunction(
        (prevCount) =>
          document.querySelectorAll('[data-testid="library-agent-card"]')
            .length > prevCount,
        initialCount,
        { timeout: 5000 },
      )
      .catch(() => {});

    const newCount = await this.getAgentCountByListLength();
    console.log(`New agent count after scroll (DOM cards): ${newCount}`);
  }

  async testPagination(): Promise<{
    initialCount: number;
    finalCount: number;
    hasMore: boolean;
  }> {
    const initialCount = await this.getAgentCountByListLength();
    await this.scrollToLoadMore();
    const finalCount = await this.getAgentCountByListLength();

    const hasMore = finalCount > initialCount;
    return {
      initialCount,
      finalCount,
      hasMore,
    };
  }

  async getAgentsWithPagination(): Promise<Agent[]> {
    console.log(`getting all agents with pagination`);

    let allAgents: Agent[] = [];
    let previousCount = 0;
    let currentCount = 0;
    const maxAttempts = 5; // Prevent infinite loop
    let attempts = 0;

    do {
      previousCount = currentCount;

      // Get current agents
      const currentAgents = await this.getAgents();
      allAgents = currentAgents;
      currentCount = currentAgents.length;

      console.log(`Attempt ${attempts + 1}: Found ${currentCount} agents`);

      // Try to load more by scrolling
      await this.scrollToLoadMore();

      attempts++;
    } while (currentCount > previousCount && attempts < maxAttempts);

    console.log(`Total agents found with pagination: ${allAgents.length}`);
    return allAgents;
  }

  async waitForPaginationLoad(): Promise<void> {
    console.log(`waiting for pagination to load`);

    // Wait for any loading states to complete
    await this.page.waitForTimeout(1000);

    // Wait for agent count to stabilize
    let previousCount = 0;
    let currentCount = 0;
    let stableChecks = 0;
    const maxChecks = 5; // Reduced from 10 to prevent excessive waiting

    while (stableChecks < 2 && stableChecks < maxChecks) {
      currentCount = await this.getAgentCount();

      if (currentCount === previousCount) {
        stableChecks++;
      } else {
        stableChecks = 0;
      }

      previousCount = currentCount;
      if (stableChecks < 2) {
        // Only wait if we haven't stabilized yet
        await this.page.waitForTimeout(500);
      }
    }

    console.log(`Pagination load stabilized with ${currentCount} agents`);
  }

  async scrollAndWaitForNewAgents(): Promise<number> {
    const initialCount = await this.getAgentCountByListLength();

    await this.scrollDown();

    await this.waitForPaginationLoad();

    const finalCount = await this.getAgentCountByListLength();
    const newAgentsLoaded = finalCount - initialCount;

    console.log(
      `Loaded ${newAgentsLoaded} new agents (${initialCount} -> ${finalCount})`,
    );

    return newAgentsLoaded;
  }

  async isPaginationWorking(): Promise<boolean> {
    const newAgentsLoaded = await this.scrollAndWaitForNewAgents();
    return newAgentsLoaded > 0;
  }
}

// Locator functions
export function getLibraryTab(page: Page): Locator {
  return page.locator('a[href="/library"]');
}

export function getAgentCards(page: Page): Locator {
  return page.getByTestId("library-agent-card");
}

export function getNewRunButton(page: Page): Locator {
  return page.getByRole("button", { name: "New run" });
}

export function getAgentTitle(page: Page): Locator {
  return page.locator("h1").first();
}

// Action functions
export async function navigateToLibrary(page: Page): Promise<void> {
  await getLibraryTab(page).click();
  await page.waitForURL(/.*\/library/);
}

export async function clickFirstAgent(page: Page): Promise<void> {
  const firstAgent = getAgentCards(page).first();
  await firstAgent.click();
}

export async function navigateToAgentByName(
  page: Page,
  agentName: string,
): Promise<void> {
  const agentCard = getAgentCards(page).filter({ hasText: agentName }).first();
  // Wait for the agent card to be visible before clicking
  // This handles async loading of agents after page navigation
  await agentCard.waitFor({ state: "visible", timeout: 15000 });
  // Click the link inside the card to navigate reliably through
  // the motion.div + draggable wrapper layers.
  const link = agentCard.locator('a[href*="/library/agents/"]').first();
  await link.click();
}

export async function clickRunButton(page: Page): Promise<void> {
  const setupTaskButton = page.getByRole("button", {
    name: /Setup your task/i,
  });
  const newTaskButton = page.getByRole("button", { name: /^New task$/i });
  const rerunTaskButton = page.getByRole("button", { name: /Rerun task/i });
  const runNowButton = page.getByRole("button", { name: /Run now/i });
  const actionButtons = [
    setupTaskButton,
    newTaskButton,
    rerunTaskButton,
    runNowButton,
  ];

  await page.waitForLoadState("domcontentloaded");
  await page.waitForLoadState("networkidle").catch(() => undefined);

  const timeoutAt = Date.now() + 20000;

  while (Date.now() < timeoutAt) {
    if (await setupTaskButton.isVisible().catch(() => false)) {
      await setupTaskButton.click();
      const startBtn = page.getByRole("button", { name: /Start Task/i }).first();
      await startBtn.waitFor({ state: "visible", timeout: 15000 });
      await fillVisibleTaskInputs(page);
      await clickStartOrSimulateTask(page, startBtn);
      return;
    }

    if (await newTaskButton.isVisible().catch(() => false)) {
      await newTaskButton.click();
      const startBtn = page.getByRole("button", { name: /Start Task/i }).first();
      await startBtn.waitFor({ state: "visible", timeout: 15000 });
      await fillVisibleTaskInputs(page);
      await clickStartOrSimulateTask(page, startBtn);
      return;
    }

    if (await rerunTaskButton.isVisible().catch(() => false)) {
      await rerunTaskButton.click();
      return;
    }

    if (await runNowButton.isVisible().catch(() => false)) {
      await runNowButton.click();
      return;
    }

    await page.waitForTimeout(250);
  }

  const visibleButtons = await page
    .getByRole("button")
    .evaluateAll((elements) =>
      elements
        .filter((element) => {
          const htmlElement = element as HTMLElement;
          const rect = htmlElement.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        })
        .map((element) => element.textContent?.trim())
        .filter(Boolean),
    );

  throw new Error(
    `Could not find run/start task button. URL: ${page.url()}. Visible buttons: ${visibleButtons.join(", ") || "none"}. Expected one of: ${actionButtons
      .map((button) => button.toString())
      .join(", ")}`,
  );
}

async function clickStartOrSimulateTask(
  page: Page,
  startBtn: Locator,
): Promise<void> {
  if (await startBtn.isEnabled()) {
    await startBtn.click({ force: true });
    await startBtn
      .waitFor({ state: "hidden", timeout: 5000 })
      .catch(async () => {
        await startBtn.click({ force: true }).catch(() => {});
      });
    return;
  }

  const simulateBtn = page.getByRole("button", { name: /Simulate/i }).first();
  if (await simulateBtn.isVisible().catch(() => false)) {
    await simulateBtn.click({ force: true });
    await simulateBtn.waitFor({ state: "hidden", timeout: 5000 }).catch(() => {
      return;
    });
    return;
  }

  throw new Error(
    "Could not start or simulate task after opening the run dialog",
  );
}

async function fillVisibleTaskInputs(page: Page): Promise<void> {
  const seededEmail = getSeededTestUser("smokeMarketplace").email;
  const inputs = page.locator(
    'input:visible:not([type="hidden"]):not([type="file"]):not([disabled]), textarea:visible:not([disabled])',
  );
  const inputCount = await inputs.count();

  for (let index = 0; index < inputCount; index += 1) {
    const input = inputs.nth(index);
    const currentValue = await input.inputValue().catch(() => "");
    if (currentValue.trim()) {
      continue;
    }

    const type = (await input.getAttribute("type"))?.toLowerCase() ?? "text";
    const placeholder = (
      (await input.getAttribute("placeholder")) ?? ""
    ).toLowerCase();
    const ariaLabel = (
      (await input.getAttribute("aria-label")) ?? ""
    ).toLowerCase();
    const labelText = `${placeholder} ${ariaLabel}`;

    if (type === "checkbox" || type === "radio") {
      continue;
    }

    const value =
      type === "email" || labelText.includes("email")
        ? seededEmail
        : type === "number"
          ? "1"
          : "e2e-input";

    await input.fill(value).catch(() => {});
  }
}

export async function clickNewRunButton(page: Page): Promise<void> {
  await getNewRunButton(page).click();
}

export async function runAgent(page: Page): Promise<void> {
  await clickRunButton(page);
}

export async function waitForAgentPageLoad(page: Page): Promise<void> {
  await page.waitForURL(/.*\/library\/agents\/[^/]+/);
  // Wait for sidebar data to finish loading so the page settles
  // into its final state (empty view vs sidebar view)
  await page.waitForLoadState("networkidle");
}

export async function getAgentName(page: Page): Promise<string> {
  return (await getAgentTitle(page).textContent()) || "";
}

export async function isLoaded(page: Page): Promise<boolean> {
  return await page.locator("h1").isVisible();
}

export async function waitForRunToComplete(
  page: Page,
  timeout = 30000,
): Promise<void> {
  await expect.poll(() => getRunStatus(page), { timeout }).not.toBe("unknown");
}

export async function getRunStatus(page: Page): Promise<string> {
  if (
    await page
      .locator(".animate-spin")
      .first()
      .isVisible()
      .catch(() => false)
  ) {
    return "running";
  }

  if (
    await getSelectors(page)
      .getId("agent-activity-badge")
      .isVisible()
      .catch(() => false)
  ) {
    return "running";
  }

  const statusTexts = [
    { text: "completed", status: "completed" },
    { text: "failed", status: "failed" },
    { text: "running", status: "running" },
    { text: "queued", status: "queued" },
    { text: "review", status: "review" },
    { text: "terminated", status: "terminated" },
    { text: "incomplete", status: "incomplete" },
  ] as const;

  for (const { text, status } of statusTexts) {
    const locator = page.getByText(new RegExp(`^${text}$`, "i")).first();
    if (await locator.isVisible().catch(() => false)) {
      return status;
    }
  }

  return "unknown";
}
