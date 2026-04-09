import { randomUUID } from "crypto";
import { expect, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

export function createUniqueAgentName(prefix: string): string {
  return `${prefix} ${Date.now()}-${randomUUID().slice(0, 8)}`;
}

export class BuildPage extends BasePage {
  // Backend-level success signal for schedule creation. Set right before we
  // click the schedule dialog's Done button so the network listener is active
  // when the POST fires. Read inside waitForScheduleCreation. Avoids a race
  // where a fast POST completes before a listener registered later attaches.
  private schedulePostPromise?: Promise<boolean>;

  constructor(page: Page) {
    super(page);
  }

  // --- Navigation ---

  async goto(): Promise<void> {
    await this.page.goto("/build");
    await this.page.waitForLoadState("domcontentloaded");
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
      await this.page
        .locator(".react-flow")
        .waitFor({ state: "visible", timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  async closeTutorial(): Promise<void> {
    try {
      await this.page
        .getByRole("button", { name: "Skip Tutorial", exact: true })
        .click({ timeout: 3000 });
    } catch {
      // Tutorial not shown or already dismissed
    }
  }

  // --- Block Menu ---

  async openBlocksPanel(): Promise<void> {
    const popoverContent = this.page.locator(
      '[data-id="blocks-control-popover-content"]',
    );
    if (!(await popoverContent.isVisible())) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
      await popoverContent.waitFor({ state: "visible", timeout: 5000 });
    }
  }

  async closeBlocksPanel(): Promise<void> {
    const popoverContent = this.page.locator(
      '[data-id="blocks-control-popover-content"]',
    );
    if (await popoverContent.isVisible()) {
      await this.page.getByTestId("blocks-control-blocks-button").click();
      await popoverContent.waitFor({ state: "hidden", timeout: 5000 });
    }
  }

  async searchBlock(searchTerm: string): Promise<void> {
    const searchInput = this.page.locator(
      '[data-id="blocks-control-search-bar"] input[type="text"]',
    );
    await searchInput.clear();
    await searchInput.fill(searchTerm);
    await expect(searchInput).toHaveValue(searchTerm);
  }

  private getBlockCardByName(name: string): Locator {
    const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const exactName = new RegExp(`^\\s*${escapedName}\\s*$`, "i");
    return this.page
      .locator('[data-id^="block-card-"]')
      .filter({ has: this.page.locator("span", { hasText: exactName }) })
      .first();
  }

  async addBlockByClick(searchTerm: string): Promise<void> {
    await this.openBlocksPanel();
    await this.searchBlock(searchTerm);

    // Wait for any search results to appear
    const anyCard = this.page.locator('[data-id^="block-card-"]').first();
    await anyCard.waitFor({ state: "visible", timeout: 10000 });

    // Click the card matching the search term name
    const blockCard = this.getBlockCardByName(searchTerm);
    await blockCard.waitFor({ state: "visible", timeout: 5000 });
    await blockCard.click();

    // Close the panel so it doesn't overlay the canvas
    await this.closeBlocksPanel();
  }

  async dragBlockToCanvas(searchTerm: string): Promise<void> {
    await this.openBlocksPanel();
    await this.searchBlock(searchTerm);

    const anyCard = this.page.locator('[data-id^="block-card-"]').first();
    await anyCard.waitFor({ state: "visible", timeout: 10000 });

    const blockCard = this.getBlockCardByName(searchTerm);
    await blockCard.waitFor({ state: "visible", timeout: 5000 });

    const canvas = this.page.locator(".react-flow__pane").first();
    await blockCard.dragTo(canvas);
  }

  // --- Nodes on Canvas ---

  getNodeLocator(index?: number): Locator {
    const locator = this.page.locator('[data-id^="custom-node-"]');
    return index !== undefined ? locator.nth(index) : locator;
  }

  async getNodeCount(): Promise<number> {
    return await this.getNodeLocator().count();
  }

  async waitForNodeOnCanvas(expectedCount?: number): Promise<void> {
    if (expectedCount !== undefined) {
      await expect(this.getNodeLocator()).toHaveCount(expectedCount, {
        timeout: 10000,
      });
    } else {
      await this.getNodeLocator()
        .first()
        .waitFor({ state: "visible", timeout: 10000 });
    }
  }

  async selectNode(index: number = 0): Promise<void> {
    const node = this.getNodeLocator(index);
    await node.click();
  }

  async selectAllNodes(): Promise<void> {
    await this.page.locator(".react-flow__pane").first().click();
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+a" : "Control+a");
  }

  async deleteSelectedNodes(): Promise<void> {
    await this.page.keyboard.press("Backspace");
  }

  // --- Connections (Edges) ---

  async connectNodes(
    sourceNodeIndex: number,
    targetNodeIndex: number,
  ): Promise<void> {
    // Get the node wrapper elements to scope handle search
    const sourceNode = this.getNodeLocator(sourceNodeIndex);
    const targetNode = this.getNodeLocator(targetNodeIndex);

    // ReactFlow renders Handle components as .react-flow__handle elements
    // Output handles have class .react-flow__handle-right (Position.Right)
    // Input handles have class .react-flow__handle-left (Position.Left)
    const sourceHandle = sourceNode
      .locator(".react-flow__handle-right")
      .first();
    const targetHandle = targetNode.locator(".react-flow__handle-left").first();

    // Get precise center coordinates using evaluate to avoid CSS transform issues
    const getHandleCenter = async (locator: Locator) => {
      const el = await locator.elementHandle();
      if (!el) throw new Error("Handle element not found");
      const rect = await el.evaluate((node) => {
        const r = node.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      });
      return rect;
    };

    const source = await getHandleCenter(sourceHandle);
    const target = await getHandleCenter(targetHandle);

    // ReactFlow requires a proper drag sequence with intermediate moves
    await this.page.mouse.move(source.x, source.y);
    await this.page.mouse.down();
    // Move in steps to trigger ReactFlow's connection detection
    const steps = 20;
    for (let i = 1; i <= steps; i++) {
      const ratio = i / steps;
      await this.page.mouse.move(
        source.x + (target.x - source.x) * ratio,
        source.y + (target.y - source.y) * ratio,
      );
    }
    await this.page.mouse.up();
  }

  async getEdgeCount(): Promise<number> {
    return await this.page.locator(".react-flow__edge").count();
  }

  // --- Save ---

  async saveAgent(
    name: string = "Test Agent",
    description: string = "",
  ): Promise<void> {
    await this.page.getByTestId("save-control-save-button").click();

    const nameInput = this.page.getByTestId("save-control-name-input");
    await nameInput.waitFor({ state: "visible", timeout: 5000 });
    await nameInput.fill(name);

    if (description) {
      await this.page
        .getByTestId("save-control-description-input")
        .fill(description);
    }

    await this.page.getByTestId("save-control-save-agent-button").click();
  }

  async waitForSaveComplete(): Promise<void> {
    await expect(this.page).toHaveURL(/flowID=/, { timeout: 15000 });
  }

  async waitForSaveButton(): Promise<void> {
    await this.page.waitForSelector(
      '[data-testid="save-control-save-button"]:not([disabled])',
      { timeout: 10000 },
    );
  }

  // --- Run ---

  async isRunButtonEnabled(): Promise<boolean> {
    const runButton = this.page.locator('[data-id="run-graph-button"]');
    return await runButton.isEnabled();
  }

  async clickRunButton(): Promise<void> {
    // Dismiss any post-save toast that may be intercepting pointer events on
    // the run button. Actively close it rather than waiting for Sonner's
    // default auto-dismiss — the auto-dismiss + fade-out routinely runs over
    // 5s and caused flakes here. The toast is optional (only after save), so
    // the dismissal is guarded.
    await this.dismissSaveToast();
    const runButton = this.page.locator('[data-id="run-graph-button"]');
    await runButton.click();
  }

  // --- Undo / Redo ---

  async isUndoEnabled(): Promise<boolean> {
    const btn = this.page.locator('[data-id="undo-button"]');
    return !(await btn.isDisabled());
  }

  async isRedoEnabled(): Promise<boolean> {
    const btn = this.page.locator('[data-id="redo-button"]');
    return !(await btn.isDisabled());
  }

  async clickUndo(): Promise<void> {
    await this.page.locator('[data-id="undo-button"]').click();
  }

  async clickRedo(): Promise<void> {
    await this.page.locator('[data-id="redo-button"]').click();
  }

  // --- Copy / Paste ---

  async copyViaKeyboard(): Promise<void> {
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+c" : "Control+c");
  }

  async pasteViaKeyboard(): Promise<void> {
    const isMac = process.platform === "darwin";
    await this.page.keyboard.press(isMac ? "Meta+v" : "Control+v");
  }

  // --- Helpers ---

  async fillBlockInputByPlaceholder(
    placeholder: string,
    value: string,
    nodeIndex: number = 0,
  ): Promise<void> {
    const node = this.getNodeLocator(nodeIndex);
    const input = node.getByPlaceholder(placeholder);
    await input.fill(value);
  }

  async clickCanvas(): Promise<void> {
    const pane = this.page.locator(".react-flow__pane").first();
    const box = await pane.boundingBox();
    if (box) {
      // Click in the center of the canvas to avoid sidebar/toolbar overlaps
      await pane.click({
        position: { x: box.width / 2, y: box.height / 2 },
      });
    } else {
      await pane.click();
    }
  }

  getPlaywrightPage(): Page {
    return this.page;
  }

  async createDummyAgent(): Promise<void> {
    await this.closeTutorial();
    await this.addBlockByClick("Add to Dictionary");
    await this.waitForNodeOnCanvas(1);
    await this.saveAgent("Test Agent", "Test Description");
    await this.waitForSaveComplete();
  }

  // --- Happy-path flows shared across PR smoke specs ---

  async open(): Promise<void> {
    await this.goto();
    await this.closeTutorial();
    await expect(this.page.locator(".react-flow")).toBeVisible({
      timeout: 15000,
    });
    await expect(
      this.page.getByTestId("blocks-control-blocks-button"),
    ).toBeVisible({ timeout: 15000 });
  }

  async addSimpleAgentBlocks(): Promise<void> {
    await this.addBlockByClick("Store Value");
    await this.waitForNodeOnCanvas(1);
    await this.fillBlockInputByPlaceholder(
      "Enter string value...",
      "smoke-value",
      0,
    );

    await this.addBlockByClick("Add to Dictionary");
    await this.waitForNodeOnCanvas(2);

    const dictionaryInputs = this.getNodeLocator(1).locator(
      'input[placeholder="Enter string value..."]',
    );
    await dictionaryInputs.nth(0).fill("smoke-key");
    await dictionaryInputs.nth(1).fill("smoke-value");

    // Connect Store Value's output to Add to Dictionary so the graph has a
    // real edge and actually produces output when run. Without this edge the
    // graph runs but emits no output, and `assertRunProducedOutput` rightly
    // fails — catching exactly the "I forgot to connect the blocks" bug
    // manual QA would catch.
    await this.connectNodes(0, 1);
  }

  async createAndSaveSimpleAgent(
    prefix: string,
  ): Promise<{ agentName: string }> {
    await this.open();
    const agentName = createUniqueAgentName(prefix);

    await this.addSimpleAgentBlocks();
    await this.saveAgent(agentName, "PR E2E builder coverage");
    await this.waitForSaveComplete();
    await this.waitForSaveButton();

    return { agentName };
  }

  async dismissSaveToast(): Promise<void> {
    const closeToastButton = this.page.getByRole("button", {
      name: "Close toast",
    });
    // Toast is optional — only shown after a save action
    if (await closeToastButton.isVisible({ timeout: 1000 })) {
      await closeToastButton.click();
    }

    // If the toast appeared but is not yet hidden, wait for it. If it never
    // appeared at all the locator is simply hidden already — no-op.
    const savedToast = this.page.getByText("Graph saved successfully");
    if (await savedToast.isVisible({ timeout: 500 })) {
      await expect(savedToast).toBeHidden({ timeout: 10000 });
    }
  }

  async startRun(): Promise<void> {
    await this.clickRunButton();

    // The run-input dialog is optional — agents without required inputs skip it
    const runDialog = this.page.locator('[data-id="run-input-dialog-content"]');
    if (await runDialog.isVisible({ timeout: 5000 })) {
      await this.page
        .locator('[data-id="run-input-manual-run-button"]')
        .click();
    }
  }

  async getExecutionState(): Promise<"running" | "idle" | "unknown"> {
    const stopButton = this.page.locator('[data-id="stop-graph-button"]');
    if (await stopButton.isVisible().catch(() => false)) {
      return "running";
    }

    const runButton = this.page.locator('[data-id="run-graph-button"]');
    if (await runButton.isVisible().catch(() => false)) {
      return "idle";
    }

    return "unknown";
  }

  // --- Tutorial (Shepherd.js tour) ---

  // Each Shepherd step's <h3> title has id="<stepId>-label"; using it avoids
  // title-overlap collisions like "Open the Block Menu" vs "The Block Menu".
  private getShepherdStep(stepId: string): Locator {
    return this.page.locator(`#${stepId}-label`);
  }

  // Scope to .shepherd-enabled so we don't click buttons on hidden-but-still-
  // attached previous steps.
  private getShepherdButton(name: string | RegExp): Locator {
    return this.page
      .locator(".shepherd-element.shepherd-enabled")
      .getByRole("button", { name });
  }

  async startTutorial(): Promise<void> {
    // Tutorial only starts from pristine /build; a flowID query param routes
    // the tutorial button to /build?view=new instead.
    await this.page.goto("/build");
    await this.page.waitForLoadState("domcontentloaded");
    await expect(this.page.locator(".react-flow")).toBeVisible({
      timeout: 15000,
    });

    await this.page.evaluate(() => {
      window.localStorage.removeItem("shepherd-tour");
    });

    const tutorialButton = this.page.locator('[data-id="tutorial-button"]');
    await expect(tutorialButton).toBeVisible({ timeout: 15000 });
    await expect(tutorialButton).toBeEnabled({ timeout: 15000 });
    await tutorialButton.click();

    await expect(this.getShepherdStep("welcome")).toBeVisible({
      timeout: 15000,
    });
  }

  async walkWelcomeToBlockMenu(): Promise<void> {
    await this.getShepherdButton("Let's Begin").click();

    await expect(this.getShepherdStep("open-block-menu")).toBeVisible({
      timeout: 10000,
    });
    await this.page
      .locator('[data-id="blocks-control-popover-trigger"]')
      .click();

    await expect(this.getShepherdStep("block-menu-overview")).toBeVisible({
      timeout: 10000,
    });
    await this.getShepherdButton("Next").click();
  }

  async walkSearchAndAddCalculator(): Promise<void> {
    // search-calculator auto-advances once the Calculator block card appears
    // in the filtered results; select-calculator auto-advances once the
    // Calculator is added to the node store.
    await expect(this.getShepherdStep("search-calculator")).toBeVisible({
      timeout: 10000,
    });
    await this.page
      .locator('[data-id="blocks-control-search-bar"] input[type="text"]')
      .fill("Calculator");

    const calculatorCard = this.page.locator(
      '[data-id="blocks-control-search-results"] [data-id="block-card-b1ab9b1967a6406dabf52dba76d00c79"]',
    );
    await expect(calculatorCard).toBeVisible({ timeout: 15000 });

    await expect(this.getShepherdStep("select-calculator")).toBeVisible({
      timeout: 15000,
    });
    await calculatorCard.scrollIntoViewIfNeeded();
    await calculatorCard.click();

    await expect(this.getShepherdStep("focus-new-block")).toBeVisible({
      timeout: 10000,
    });
    await this.waitForNodeOnCanvas(1);
  }

  // Use dispatchEvent — the Shepherd cancel icon sits inside a step that's
  // pinned to an off-screen React Flow node, so Playwright's visibility
  // checks reject a normal click. A synthetic click event still triggers
  // tour.cancel() via Shepherd's listener.
  async cancelTutorial(): Promise<void> {
    await this.page
      .locator(".shepherd-element.shepherd-enabled .shepherd-cancel-icon")
      .first()
      .dispatchEvent("click");
    await expect(
      this.page.locator(".shepherd-element.shepherd-enabled"),
    ).toHaveCount(0, { timeout: 10000 });
  }

  // NOTE: welcome.ts "Skip Tutorial" only calls handleTutorialSkip, which
  // writes localStorage but does NOT call tour.cancel(). The tour UI stays
  // open — the skip state is persisted so the next /build visit knows the
  // user already dismissed the tour. Callers that want the UI closed must
  // also call cancelTutorial().
  async skipTutorialFromWelcome(): Promise<void> {
    await expect(this.getShepherdStep("welcome")).toBeVisible({
      timeout: 10000,
    });
    await this.getShepherdButton(/Skip Tutorial/i).click();
    await expect
      .poll(() => this.getTutorialStateFromStorage(), { timeout: 5000 })
      .toBe("skipped");
  }

  async getTutorialStateFromStorage(): Promise<string | null> {
    return this.page.evaluate(() =>
      window.localStorage.getItem("shepherd-tour"),
    );
  }

  // --- Scheduling ---

  private async waitForScheduleUi(): Promise<{
    state: "ready" | "pending";
    runDialog: Locator;
    scheduleDialog: Locator;
  }> {
    const runDialog = this.page.locator('[data-id="run-input-dialog-content"]');
    const scheduleDialog = this.page.getByRole("dialog", {
      name: "Schedule Graph",
    });

    const state = await expect
      .poll(
        async () => {
          if (await scheduleDialog.isVisible().catch(() => false)) {
            return "schedule";
          }
          if (await runDialog.isVisible().catch(() => false)) {
            return "run-input";
          }
          return "pending";
        },
        { timeout: 8000 },
      )
      .not.toBe("pending")
      .then(() => "ready" as const)
      .catch(() => "pending" as const);

    return { state, runDialog, scheduleDialog };
  }

  async openScheduleDialog(): Promise<{
    runDialog: Locator;
    scheduleDialog: Locator;
  }> {
    const scheduleButton = this.page.locator(
      '[data-id="schedule-graph-button"]',
    );

    await this.dismissSaveToast();

    for (let attempt = 0; attempt < 2; attempt++) {
      await expect(scheduleButton).toBeVisible({ timeout: 15000 });
      await expect(scheduleButton).toBeEnabled({ timeout: 15000 });
      await scheduleButton.click();

      const { state, runDialog, scheduleDialog } =
        await this.waitForScheduleUi();
      if (state !== "pending") {
        return { runDialog, scheduleDialog };
      }

      await this.page.reload();
      await this.page.waitForLoadState("domcontentloaded");
      await this.closeTutorial();
      await expect(this.page.locator(".react-flow")).toBeVisible({
        timeout: 15000,
      });
      await this.dismissSaveToast();
    }

    throw new Error("Schedule UI did not open from the builder");
  }

  private async configureSchedule(): Promise<void> {
    const hourSelect = this.page.locator("#time-hour");
    await expect(hourSelect).toBeVisible({ timeout: 15000 });

    const currentHourText = (await hourSelect.textContent()) ?? "";
    const currentHourMatch = currentHourText.match(/\b(1[0-2]|[1-9])\b/);
    const currentHour = currentHourMatch?.[0] ?? "9";
    const nextHour = currentHour === "10" ? "11" : "10";

    await hourSelect.click();

    const nextHourOption = this.page.getByRole("option", {
      name: nextHour,
      exact: true,
    });
    await nextHourOption.waitFor({ state: "visible", timeout: 15000 });
    await nextHourOption.click();

    await expect(hourSelect).toContainText(nextHour);
  }

  private async waitForScheduleCreation(
    scheduleDialog: Locator,
  ): Promise<void> {
    const successToastTitle = this.page.getByText("Schedule created", {
      exact: true,
    });
    const successToastDescription = this.page.getByText(
      "Schedule created successfully",
    );
    const invalidScheduleToast = this.page.getByText("Invalid schedule", {
      exact: true,
    });
    const failedScheduleToast = this.page.getByText(
      "Failed to create schedule",
      {
        exact: true,
      },
    );

    // Consume the network-level success signal that was registered BEFORE
    // the Done click in createScheduleForSavedAgent. If it resolves true, the
    // backend created the schedule even if the UI dialog never transitioned.
    let schedulePostSucceeded = false;
    const postPromise = this.schedulePostPromise;
    if (postPromise) {
      postPromise.then((ok) => {
        schedulePostSucceeded = ok;
      });
    }

    await expect
      .poll(
        async () => {
          if (
            (await successToastTitle.isVisible().catch(() => false)) ||
            (await successToastDescription.isVisible().catch(() => false))
          ) {
            return "success";
          }

          if (!(await scheduleDialog.isVisible().catch(() => false))) {
            return "success";
          }

          if (schedulePostSucceeded) {
            return "success";
          }

          if (await invalidScheduleToast.isVisible().catch(() => false)) {
            return "invalid";
          }

          if (await failedScheduleToast.isVisible().catch(() => false)) {
            return "failed";
          }

          return "pending";
        },
        { timeout: 120000 },
      )
      .toBe("success");
  }

  async createScheduleForSavedAgent(agentName: string): Promise<void> {
    const { runDialog, scheduleDialog } = await this.openScheduleDialog();

    if (
      (await runDialog.isVisible({ timeout: 1000 }).catch(() => false)) &&
      !(await scheduleDialog.isVisible({ timeout: 1000 }).catch(() => false))
    ) {
      await this.page.locator('[data-id="run-input-schedule-button"]').click();
    }

    await expect(scheduleDialog).toBeVisible({ timeout: 15000 });
    await this.page.locator("#schedule-name").fill(`Daily ${agentName}`);
    await this.configureSchedule();

    const doneButton = scheduleDialog.getByRole("button", {
      name: "Done",
      exact: true,
    });
    await expect(doneButton).toBeEnabled({ timeout: 15000 });

    // CRITICAL: register the network listener BEFORE clicking Done, otherwise
    // the POST can fire and return before the listener attaches, causing a
    // 120s false-pending timeout even though the schedule was created.
    // waitForScheduleCreation reads this promise.
    this.schedulePostPromise = this.page
      .waitForResponse(
        (res) =>
          res.request().method() === "POST" &&
          /\/api\/graphs\/[^/]+\/schedules/.test(res.url()) &&
          res.status() >= 200 &&
          res.status() < 300,
        { timeout: 120000 },
      )
      .then(() => true)
      .catch(() => false);

    const getScheduleSubmissionState = async () => {
      if (!(await scheduleDialog.isVisible().catch(() => false))) {
        return "submitted";
      }

      if (
        await scheduleDialog
          .getByRole("button", { name: /Creating schedule/i })
          .isVisible()
          .catch(() => false)
      ) {
        return "submitted";
      }

      return "idle";
    };

    await doneButton.click();

    const initialSubmissionStarted = await expect
      .poll(getScheduleSubmissionState, { timeout: 5000 })
      .not.toBe("idle")
      .then(() => true)
      .catch(() => false);

    if (!initialSubmissionStarted) {
      await doneButton.click();
      await expect
        .poll(getScheduleSubmissionState, { timeout: 5000 })
        .not.toBe("idle");
    }

    await this.waitForScheduleCreation(scheduleDialog);
    await expect(scheduleDialog).toBeHidden({ timeout: 15000 });
  }
}
