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

  async openSavedAgent(agentName: string): Promise<void> {
    await openSavedAgentInLibrary(this.page, agentName);
  }

  async waitForRunToComplete(timeout = 45000): Promise<void> {
    await waitForRunToComplete(this.page, timeout);
  }

  async getRunStatus(): Promise<string> {
    return getRunStatus(this.page);
  }

  async assertRunProducedOutput(timeout = 15000): Promise<void> {
    await assertRunProducedOutput(this.page, timeout);
  }

  async assertRunOutputValue(
    outputName: string,
    expectedValue: RegExp | string,
    timeout = 15000,
  ): Promise<void> {
    await assertRunOutputValue(this.page, outputName, expectedValue, timeout);
  }

  async assertFirstRunOutputValue(
    expectedValue: RegExp | string,
    timeout = 15000,
  ): Promise<void> {
    await assertRunOutputContainsText(this.page, expectedValue, timeout);
  }

  async clickExportAgent(): Promise<void> {
    await clickExportAgent(this.page);
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
    await expect(searchInput).toHaveValue(searchTerm);
  }

  async clearSearch(): Promise<void> {
    console.log(`clearing search`);
    // Look for the clear button (X icon)
    const clearButton = this.page.locator(".lucide.lucide-x");
    const searchInput = this.page.getByRole("textbox", {
      name: "Search agents",
    });
    if (await clearButton.isVisible()) {
      await clearButton.click();
    } else {
      // If no clear button, clear the search input directly
      await searchInput.fill("");
    }
    await expect(searchInput).toHaveValue("");
  }

  async selectSortOption(
    page: Page,
    sortOption: "Creation Date" | "Last Modified",
  ): Promise<void> {
    const { getRole } = getSelectors(page);
    await getRole("combobox").click();

    await getRole("option", sortOption).click();
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

    const agentCard = this.page
      .getByTestId("library-agent-card")
      .filter({ hasText: agent.name });

    // The "Edit agent" link is inside the three-dot dropdown menu.
    // Open the menu first, then click the builder link.
    const menuTrigger = agentCard.getByRole("button", {
      name: "More actions",
    });
    await menuTrigger.first().click();

    const builderLink = this.page.getByTestId(
      "library-agent-card-open-in-builder-link",
    );
    await builderLink.first().click();
  }

  async waitForAgentsToLoad(): Promise<void> {
    const { getId } = getSelectors(this.page);
    await expect
      .poll(
        async () => {
          const [agentCardVisible, agentsCountVisible] = await Promise.all([
            getId("library-agent-card")
              .first()
              .isVisible()
              .catch(() => false),
            getId("agents-count")
              .isVisible()
              .catch(() => false),
          ]);

          return agentCardVisible || agentsCountVisible;
        },
        { timeout: 10_000 },
      )
      .toBe(true);
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
    return noAgentsText.isVisible();
  }

  async scrollToBottom(): Promise<void> {
    console.log(`scrolling to bottom to trigger pagination`);
    await this.page.keyboard.press("End");
  }

  async scrollDown(): Promise<void> {
    console.log(`scrolling down to trigger pagination`);
    await this.page.keyboard.press("PageDown");
  }

  // Returns true if more agents loaded, false if we're on the last page.
  // Callers must distinguish these cases so a broken pagination pipeline
  // doesn't quietly look like "we reached the end".
  async scrollToLoadMore(): Promise<boolean> {
    const initialCount = await this.getAgentCountByListLength();
    console.log(`Initial agent count (DOM cards): ${initialCount}`);

    await this.scrollToBottom();

    try {
      await this.page.waitForFunction(
        (prevCount) =>
          document.querySelectorAll('[data-testid="library-agent-card"]')
            .length > prevCount,
        initialCount,
        { timeout: 10000 },
      );
      return true;
    } catch {
      // No new cards — caller should verify this is actually the last page
      // (e.g., by comparing against `getAgentCount()`), not a broken fetch.
      return false;
    }
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
    // Wait until the agent count header stops changing. Poll every 500ms
    // and declare stable after two consecutive equal reads, capped at 10s.
    // The previous implementation had no delay between reads and so hit
    // "stable" instantly — effectively a no-op.
    const deadline = Date.now() + 10000;
    let previousCount = -1;
    let stableChecks = 0;

    while (Date.now() < deadline && stableChecks < 2) {
      const currentCount = await this.getAgentCount();
      if (currentCount === previousCount) {
        stableChecks += 1;
      } else {
        stableChecks = 0;
        previousCount = currentCount;
      }
      await this.page.waitForTimeout(500);
    }
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
    if (
      await setupTaskButton
        .first()
        .isVisible()
        .catch(() => false)
    ) {
      const clicked = await clickActionButton(setupTaskButton.first());
      if (!clicked) {
        await page.waitForTimeout(250);
        continue;
      }

      const runDialog = await waitForRunDialog(page);
      await fillVisibleTaskInputs(runDialog);
      await clickStartOrSimulateTask(page, runDialog);
      return;
    }

    if (
      await newTaskButton
        .first()
        .isVisible()
        .catch(() => false)
    ) {
      const clicked = await clickActionButton(newTaskButton.first());
      if (!clicked) {
        await page.waitForTimeout(250);
        continue;
      }

      const runDialog = await waitForRunDialog(page);
      await fillVisibleTaskInputs(runDialog);
      await clickStartOrSimulateTask(page, runDialog);
      return;
    }

    if (
      await rerunTaskButton
        .first()
        .isVisible()
        .catch(() => false)
    ) {
      const clicked = await clickActionButton(rerunTaskButton.first());
      if (!clicked) {
        await page.waitForTimeout(250);
        continue;
      }

      return;
    }

    if (
      await runNowButton
        .first()
        .isVisible()
        .catch(() => false)
    ) {
      const clicked = await clickActionButton(runNowButton.first());
      if (!clicked) {
        await page.waitForTimeout(250);
        continue;
      }

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

async function clickActionButton(button: Locator): Promise<boolean> {
  try {
    await expect(button).toBeVisible({ timeout: 2000 });
    await expect(button).toBeEnabled({ timeout: 2000 });
    await button.click({ timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

async function waitForRunDialog(page: Page): Promise<Locator> {
  const runDialog = page
    .locator("[data-dialog-content]")
    .filter({
      has: page.getByRole("button", { name: /^Start Task$/i }),
    })
    .last();
  await expect(runDialog).toBeVisible({ timeout: 15000 });
  return runDialog;
}

async function dismissRunSafetyPopup(page: Page): Promise<void> {
  const safetyPopup = page
    .locator("[data-dialog-content]")
    .filter({
      has: page.getByText("Safety Checks Enabled", { exact: true }),
    })
    .last();

  if (!(await safetyPopup.isVisible({ timeout: 2000 }).catch(() => false))) {
    return;
  }

  await safetyPopup.getByRole("button", { name: /^Got it$/i }).click();
  await expect(safetyPopup).toBeHidden({ timeout: 10000 });
}

async function clickStartOrSimulateTask(
  page: Page,
  runDialog: Locator,
): Promise<void> {
  const startBtn = runDialog.getByRole("button", { name: /^Start Task$/i });
  // Happy-path tests must exercise a real run — do NOT fall back to the
  // "Simulate" button if Start fails, because a broken Start code path is
  // exactly the regression these tests exist to catch.
  await expect(startBtn).toBeVisible({ timeout: 10000 });
  await expect(startBtn).toBeEnabled({ timeout: 10000 });
  await startBtn.click();
  await dismissRunSafetyPopup(page);

  await expect
    .poll(
      () => {
        const currentUrl = new URL(page.url());
        return (
          currentUrl.searchParams.get("activeTab") === "runs" &&
          currentUrl.searchParams.get("activeItem") !== null
        );
      },
      {
        timeout: 15000,
        message:
          "Start Task click did not navigate to a run detail (?activeTab=runs&activeItem=...)",
      },
    )
    .toBe(true);
}

async function fillVisibleTaskInputs(container: Page | Locator): Promise<void> {
  const seededEmail = getSeededTestUser("smokeMarketplace").email;
  const inputs = container.locator(
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
    const inputMetadata = await input.evaluate((element) => {
      const formField = element as HTMLInputElement | HTMLTextAreaElement;
      const closestLabel = formField.closest("label")?.textContent ?? "";
      const forLabel = formField.id
        ? (document.querySelector(`label[for="${CSS.escape(formField.id)}"]`)
            ?.textContent ?? "")
        : "";

      return {
        placeholder: formField.getAttribute("placeholder") ?? "",
        ariaLabel: formField.getAttribute("aria-label") ?? "",
        name: formField.getAttribute("name") ?? "",
        labelText: `${closestLabel} ${forLabel}`.trim(),
      };
    });
    const fieldDescriptor = [
      inputMetadata.placeholder,
      inputMetadata.ariaLabel,
      inputMetadata.name,
      inputMetadata.labelText,
    ]
      .join(" ")
      .toLowerCase();

    if (type === "checkbox" || type === "radio") {
      continue;
    }

    const value =
      type === "email" || fieldDescriptor.includes("email")
        ? seededEmail
        : type === "number" ||
            /\b(a|b)\b/.test(fieldDescriptor) ||
            fieldDescriptor.includes("number")
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

export async function waitForAgentPageLoad(
  page: Page,
  agentName?: string,
): Promise<void> {
  await page.waitForURL(/.*\/library\/agents\/[^/]+/);
  // Wait for the primary content area to be present so the page has settled
  // into its final state (empty view vs sidebar view)
  await page.waitForLoadState("domcontentloaded");

  // Transient "Something went wrong — All connection attempts failed" error
  // boundary appears when the library agent page loads before the backend
  // has indexed a newly-cloned agent (race between marketplace "Add to
  // Library" and backend availability). Click "Try Again" and re-settle.
  const errorHeading = page.getByText("Something went wrong", {
    exact: false,
  });
  let errorResolved = false;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (!(await errorHeading.isVisible({ timeout: 300 }).catch(() => false))) {
      errorResolved = true;
      break;
    }
    const tryAgain = page.getByRole("button", { name: "Try Again" });
    if (await tryAgain.isVisible({ timeout: 500 }).catch(() => false)) {
      await tryAgain.click();
    } else {
      await page.reload();
    }
    await page.waitForLoadState("domcontentloaded");
  }

  if (!errorResolved) {
    errorResolved = !(await errorHeading
      .isVisible({ timeout: 300 })
      .catch(() => false));
  }

  if (!errorResolved) {
    throw new Error(
      "Library agent page remained on the connection-failure screen after 3 retries",
    );
  }

  await waitForAgentDetailShell(page, agentName);
}

async function waitForLibraryListToLeave(page: Page): Promise<void> {
  const librarySearch = page.getByTestId("library-textbox");
  await expect
    .poll(
      async () => {
        const count = await librarySearch.count();
        if (count === 0) {
          return "gone";
        }

        if (
          !(await librarySearch
            .first()
            .isVisible()
            .catch(() => false))
        ) {
          return "gone";
        }

        return "visible";
      },
      { timeout: 15000 },
    )
    .toBe("gone");
}

async function getVisibleAgentDetailSurface(page: Page): Promise<string> {
  const visibleSurfaces: Array<[string, Locator]> = [
    [
      "about-agent",
      page.getByText("About this agent", { exact: true }).first(),
    ],
    [
      "setup-task",
      page.getByRole("button", { name: /^Setup your task$/i }).first(),
    ],
    ["new-task", page.getByRole("button", { name: /^New task$/i }).first()],
    ["scheduled-tab", page.getByRole("tab", { name: /^Scheduled$/i }).first()],
  ];

  for (const [surface, locator] of visibleSurfaces) {
    if (await locator.isVisible().catch(() => false)) {
      return surface;
    }
  }

  return "pending";
}

async function waitForAgentDetailShell(
  page: Page,
  agentName?: string,
): Promise<void> {
  await waitForLibraryListToLeave(page);

  await expect(
    page.getByRole("link", { name: "My Library" }).first(),
  ).toBeVisible({
    timeout: 15000,
  });

  if (agentName) {
    await expect(
      page
        .locator(`a[href*="/library/agents/"]`)
        .filter({ hasText: agentName })
        .first(),
    ).toBeVisible({ timeout: 15000 });
  }

  await expect
    .poll(() => getVisibleAgentDetailSurface(page), { timeout: 15000 })
    .not.toBe("pending");
}

export async function getAgentName(page: Page): Promise<string> {
  return (await getAgentTitle(page).textContent()) || "";
}

export async function isLoaded(page: Page): Promise<boolean> {
  return await page.locator("h1").isVisible();
}

const SUCCESS_RUN_STATUS = "completed";
const FAILURE_RUN_STATUSES = new Set(["failed", "terminated", "incomplete"]);
const RUN_ERROR_RECOVERY_GRACE_PERIOD_MS = 1500;
const RUN_ERROR_RECOVERY_ATTEMPTS = 2;

/**
 * Assert that a completed run actually produced output.
 *
 * The Library run-detail Output panel renders "No output from this run." when
 * the run object has no `outputs` field. There's a brief window after the run
 * reaches "completed" status where the run object is loaded without outputs,
 * then outputs arrive and the panel re-renders. We poll for up to `timeout`
 * ms waiting for the "No output" placeholder to GO AWAY before concluding
 * the run genuinely produced nothing.
 *
 * This catches the "agent runs but produces nothing" failure mode
 * (disconnected edges, broken graph, runtime crash before any output node
 * fired) — the exact regression that ACCEPTED_RUN_STATUSES previously hid.
 */
export async function assertRunProducedOutput(
  page: Page,
  timeout = 15000,
): Promise<void> {
  await openRunOutputTab(page);

  // A completed run must surface output on the CURRENT render without a
  // page reload. Reloading to "rule out stale cache" would mask a real
  // user-visible regression where the frontend only shows output after a
  // manual refresh.
  const noOutput = page.getByText("No output from this run.", { exact: true });
  await expect(noOutput, {
    message:
      'run completed but produced no output ("No output from this run." still shown) — broken graph, missing output node, or stale React Query cache',
  }).toBeHidden({ timeout });
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function openRunOutputTab(page: Page): Promise<void> {
  const outputTab = page.getByRole("tab", { name: /^Output$/i }).first();
  if (await outputTab.isVisible().catch(() => false)) {
    await outputTab.click();
    return;
  }

  const outputButton = page.getByRole("button", { name: /^Output$/i }).first();
  if (await outputButton.isVisible().catch(() => false)) {
    await outputButton.click();
  }
}

export async function assertRunOutputValue(
  page: Page,
  outputName: string,
  expectedValue: RegExp | string,
  timeout = 15000,
): Promise<void> {
  await openRunOutputTab(page);

  const outputLabel = page.locator("p.capitalize:visible").filter({
    hasText: new RegExp(`^${escapeRegex(outputName)}$`, "i"),
  });

  await expect(
    outputLabel,
    `run output should include output key "${outputName}"`,
  ).toBeVisible({ timeout });

  const outputValue = outputLabel.locator("xpath=following-sibling::*[1]");
  if (expectedValue instanceof RegExp) {
    await expect(
      outputValue,
      `run output value for "${outputName}" should match ${expectedValue.toString()}`,
    ).toHaveText(expectedValue, { timeout });
    return;
  }

  await expect(
    outputValue,
    `run output value for "${outputName}" should be "${expectedValue}"`,
  ).toHaveText(expectedValue, { timeout });
}

export async function assertFirstRunOutputValue(
  page: Page,
  expectedValue: RegExp | string,
  timeout = 15000,
): Promise<void> {
  await assertRunOutputContainsText(page, expectedValue, timeout);
}

export async function assertRunOutputContainsText(
  page: Page,
  expectedValue: RegExp | string,
  timeout = 15000,
): Promise<void> {
  await openRunOutputTab(page);

  const outputCard = page
    .locator("div")
    .filter({
      has: page.getByRole("button", { name: "Copy all text outputs" }),
    })
    .first();
  await expect(outputCard, "run output card should be visible").toBeVisible({
    timeout,
  });

  if (expectedValue instanceof RegExp) {
    await expect(
      outputCard.getByText(expectedValue).first(),
      `run output should contain text matching ${expectedValue.toString()}`,
    ).toBeVisible({ timeout });
    return;
  }

  await expect(
    outputCard.getByText(expectedValue, { exact: true }).first(),
    `run output should contain "${expectedValue}"`,
  ).toBeVisible({ timeout });
}

export async function waitForRunToComplete(
  page: Page,
  timeout = 45000,
): Promise<void> {
  const start = Date.now();
  let lastStatus = "unknown";
  let runErrorDetectedAt: number | null = null;
  let recoveryAttempts = 0;
  while (Date.now() - start < timeout) {
    lastStatus = await getRunStatus(page);
    if (lastStatus === SUCCESS_RUN_STATUS) {
      return;
    }
    if (lastStatus === "error") {
      runErrorDetectedAt ??= Date.now();
      if (
        Date.now() - runErrorDetectedAt >=
        RUN_ERROR_RECOVERY_GRACE_PERIOD_MS
      ) {
        if (recoveryAttempts >= RUN_ERROR_RECOVERY_ATTEMPTS) {
          throw new Error(`Run reached terminal failure state "${lastStatus}"`);
        }
        recoveryAttempts += 1;
        runErrorDetectedAt = null;
        await page.reload();
        await waitForAgentPageLoad(page);
        continue;
      }
    } else {
      runErrorDetectedAt = null;
    }
    if (FAILURE_RUN_STATUSES.has(lastStatus)) {
      throw new Error(`Run reached terminal failure state "${lastStatus}"`);
    }
    await page.waitForTimeout(250);
  }
  throw new Error(
    `waitForRunToComplete timed out after ${timeout}ms — last status was "${lastStatus}" (expected "${SUCCESS_RUN_STATUS}")`,
  );
}

export function getActiveItemId(page: Page): string | null {
  return new URL(page.url()).searchParams.get("activeItem");
}

export async function dismissFeedbackDialog(page: Page): Promise<void> {
  const feedbackDialog = page.getByRole("dialog", {
    name: "We'd love your feedback",
  });
  // Dialog is genuinely optional — it only appears on some run completions.
  // Give it a realistic window to animate in; 500ms races the dialog
  // transition and causes later clicks to land on it instead of the button
  // behind it.
  if (!(await feedbackDialog.isVisible({ timeout: 3000 }).catch(() => false))) {
    return;
  }

  const cancelButton = feedbackDialog.getByRole("button", { name: "Cancel" });
  if (await cancelButton.isVisible()) {
    await cancelButton.click();
    await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
    return;
  }

  await feedbackDialog.getByRole("button", { name: "Close" }).click();
  await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
}

export async function importAgentFromFile(
  page: Page,
  filePath: string,
  agentName: string,
  description: string = "PR E2E library coverage",
): Promise<{ libraryPage: LibraryPage; importedAgent: Agent }> {
  const libraryPage = new LibraryPage(page);
  const importDialog = page.getByRole("dialog", { name: "Import" });

  await page.goto("/library");
  await libraryPage.openUploadDialog();
  await libraryPage.fillUploadForm(agentName, description);

  const fileInput = importDialog.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);
  const uploadButton = importDialog.getByRole("button", { name: "Upload" });
  await expect(uploadButton).toBeEnabled({
    timeout: 10000,
  });
  await uploadButton.click();
  const uploadingButton = importDialog.getByRole("button", {
    name: /Uploading\.\.\./i,
  });
  const sawUploadingState = await uploadingButton
    .waitFor({ state: "visible", timeout: 2000 })
    .then(() => true)
    .catch(() => false);
  if (sawUploadingState) {
    await expect
      .poll(
        async () => {
          if (/\/build/.test(page.url())) {
            return "build";
          }
          if (!(await uploadingButton.isVisible().catch(() => false))) {
            return "gone";
          }
          return (await uploadingButton.isDisabled().catch(() => false))
            ? "disabled"
            : "enabled";
        },
        {
          timeout: 5000,
          message:
            'upload button should either stay disabled while "Uploading..." is visible or disappear because navigation already started',
        },
      )
      .not.toBe("enabled");
  }

  // Upload → backend creates the graph → router pushes /build?flowID=...
  // This pipeline includes file parsing plus a backend graph creation call.
  // On a cold stack it can take longer than a normal UI transition, so poll
  // for the real terminal states: builder navigation or an explicit error.
  await expect
    .poll(
      async () => {
        if (/\/build/.test(page.url())) {
          return "build";
        }

        const uploadFailed = await page
          .getByText("Error Uploading agent")
          .isVisible()
          .catch(() => false);
        if (uploadFailed) {
          return "failed";
        }

        return "pending";
      },
      {
        timeout: 60000,
        message:
          "agent import should either navigate to /build or surface an explicit upload error toast",
      },
    )
    .toBe("build");
  await expect(page).toHaveURL(/\/build/, { timeout: 15000 });

  // Import should produce a real graph, not an empty canvas. Lazy-import
  // BuildPage locally to avoid a circular dependency between the two
  // page-object modules.
  const { BuildPage } = await import("./build.page");
  const importedBuildPage = new BuildPage(page);
  await importedBuildPage.waitForNodeOnCanvas();
  const importedNodeCount = await importedBuildPage.getNodeCount();
  expect(
    importedNodeCount,
    "imported agent must render at least one node on canvas",
  ).toBeGreaterThan(0);

  await page.goto("/library");
  await libraryPage.searchAgents(agentName);
  await libraryPage.waitForAgentsToLoad();

  // Look up the specific imported card directly rather than calling
  // getAgents() in a loop. getAgents() iterates every visible card and
  // reads hrefs via `.getAttribute`, which deadlocks if the library list
  // re-renders mid-iteration (previously caused this test to hang 120s on
  // the 8th card). A filter-based lookup on the agent name is both faster
  // and immune to list churn.
  const { getId } = getSelectors(page);
  const importedCard = getId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(
    importedCard,
    `imported agent card "${agentName}" must appear in the library search results`,
  ).toBeVisible({ timeout: 15000 });

  const seeRunsLink = getId("library-agent-card-see-runs-link", importedCard);
  const seeRunsUrl = (await seeRunsLink.getAttribute("href")) ?? "";
  const openInBuilderLink = getId(
    "library-agent-card-open-in-builder-link",
    importedCard,
  );
  const openInBuilderUrl =
    (await openInBuilderLink.count()) > 0
      ? ((await openInBuilderLink.getAttribute("href")) ?? "")
      : "";

  const idMatch = seeRunsUrl.match(/\/library\/agents\/([^/]+)/);
  const importedAgent: Agent = {
    id: idMatch ? idMatch[1] : "",
    name:
      (
        await getId("library-agent-card-name", importedCard).textContent()
      )?.trim() ?? agentName,
    description: "",
    seeRunsUrl,
    openInBuilderUrl,
  };

  expect(
    importedAgent.name,
    "imported agent name should contain the requested name",
  ).toContain(agentName);

  return { libraryPage, importedAgent };
}

export async function openSavedAgentInLibrary(
  page: Page,
  agentName: string,
): Promise<void> {
  const libraryPage = new LibraryPage(page);

  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);
  await libraryPage.waitForAgentsToLoad();
  await navigateToAgentByName(page, agentName);
  await waitForAgentPageLoad(page, agentName);
}

async function waitForExportActionSurface(
  page: Page,
): Promise<"direct" | "menu"> {
  await expect
    .poll(
      async () => {
        if (
          await getFirstVisibleLocator(page, "button", "Export agent to file")
        ) {
          return "direct";
        }

        if (await getFirstVisibleLocator(page, "button", "More actions")) {
          return "menu";
        }

        return "pending";
      },
      { timeout: 30000 },
    )
    .not.toBe("pending");

  if (await getFirstVisibleLocator(page, "button", "Export agent to file")) {
    return "direct";
  }

  return "menu";
}

async function getFirstVisibleLocator(
  page: Page,
  role: "button" | "menuitem",
  name: string,
): Promise<Locator | null> {
  const locator = page.getByRole(role, { name });
  const count = await locator.count();

  for (let index = 0; index < count; index += 1) {
    const candidate = locator.nth(index);
    if (await candidate.isVisible().catch(() => false)) {
      return candidate;
    }
  }

  return null;
}

export async function clickExportAgent(page: Page): Promise<void> {
  const exportSurface = await waitForExportActionSurface(page);

  if (exportSurface === "direct") {
    const directExportButton = await getFirstVisibleLocator(
      page,
      "button",
      "Export agent to file",
    );
    if (!directExportButton) {
      throw new Error(
        "Export button was not visible after export surface resolved",
      );
    }

    await directExportButton.click({ timeout: 15000 });
    return;
  }

  const moreActionsButtons = page.getByRole("button", { name: "More actions" });
  const moreActionsCount = await moreActionsButtons.count();

  for (let index = 0; index < moreActionsCount; index += 1) {
    const moreActionsButton = moreActionsButtons.nth(index);

    if (!(await moreActionsButton.isVisible().catch(() => false))) {
      continue;
    }

    await moreActionsButton.click({ timeout: 15000 });

    const exportMenuItem = await getFirstVisibleLocator(
      page,
      "menuitem",
      "Export agent to file",
    );
    if (exportMenuItem) {
      await exportMenuItem.click({ timeout: 15000 });
      return;
    }

    await page.keyboard.press("Escape").catch(() => {});
  }

  throw new Error(
    "Export action was not available from any visible More actions menu",
  );
}

// The run status is rendered by RunStatusBadge as lowercase text inside a
// `.capitalize` element (uppercased via CSS). Scoping to that class prevents
// false positives from free-text occurrences of words like "completed"
// elsewhere on the page (filter chips, tooltips, etc.).
const RUN_STATUS_WORDS = [
  "completed",
  "failed",
  "terminated",
  "incomplete",
  "queued",
  "review",
  "running",
] as const;

export async function getRunStatus(page: Page): Promise<string> {
  // 1. Detect React error boundary first — fast loud failure if the page
  //    crashed mid-run, instead of polling until timeout.
  const errorBoundary = page.getByText(
    /Something went wrong|We had the following error|Application error/i,
  );
  if (
    await errorBoundary
      .first()
      .isVisible({ timeout: 200 })
      .catch(() => false)
  ) {
    return "error";
  }

  // 2. Read the status from the scoped RunStatusBadge element. This is the
  //    only source of truth — no free-text matching across the whole page,
  //    no spinner heuristics that confuse a skeleton loader with a live run.
  const badges = page.locator(".capitalize");
  const badgeCount = await badges.count().catch(() => 0);
  for (let i = 0; i < badgeCount; i += 1) {
    const badge = badges.nth(i);
    if (!(await badge.isVisible().catch(() => false))) continue;
    const text = ((await badge.textContent()) ?? "").trim().toLowerCase();
    if ((RUN_STATUS_WORDS as readonly string[]).includes(text)) {
      return text;
    }
  }

  return "unknown";
}
