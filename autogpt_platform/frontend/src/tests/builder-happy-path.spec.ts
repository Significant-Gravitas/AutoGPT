import { Locator, Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import {
  navigateToAgentByName,
  waitForAgentPageLoad,
} from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.builder });

function createUniqueAgentName(prefix: string) {
  return `${prefix} ${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function openBuilder(page: Page) {
  const buildPage = new BuildPage(page);

  await page.goto("/build");
  await page.waitForLoadState("domcontentloaded");
  await buildPage.closeTutorial();

  await expect(page.locator(".react-flow")).toBeVisible();
  await expect(page.getByTestId("blocks-control-blocks-button")).toBeVisible();

  return buildPage;
}

async function addSimpleAgentBlocks(buildPage: BuildPage) {
  await buildPage.addBlockByClick("Store Value");
  await buildPage.waitForNodeOnCanvas(1);
  await buildPage.fillBlockInputByPlaceholder(
    "Enter string value...",
    "smoke-value",
    0,
  );

  await buildPage.addBlockByClick("Add to Dictionary");
  await buildPage.waitForNodeOnCanvas(2);

  const dictionaryInputs = buildPage
    .getNodeLocator(1)
    .locator('input[placeholder="Enter string value..."]');
  await dictionaryInputs.nth(0).fill("smoke-key");
  await dictionaryInputs.nth(1).fill("smoke-value");
}

async function createAndSaveAgent(page: Page, prefix: string) {
  const buildPage = await openBuilder(page);
  const agentName = createUniqueAgentName(prefix);

  await addSimpleAgentBlocks(buildPage);
  await buildPage.saveAgent(agentName, "PR E2E builder coverage");
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  return { buildPage, agentName };
}

async function dismissSaveToast(page: Page) {
  const closeToastButton = page.getByRole("button", { name: "Close toast" });
  if (await closeToastButton.isVisible({ timeout: 1000 }).catch(() => false)) {
    await closeToastButton.click();
  }

  await page
    .getByText("Graph saved successfully")
    .waitFor({ state: "hidden", timeout: 10000 })
    .catch(() => {});
}

async function waitForScheduleUi(page: Page) {
  const runDialog = page.locator('[data-id="run-input-dialog-content"]');
  const scheduleDialog = page.getByRole("dialog", { name: "Schedule Graph" });

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
    .then(() => "ready")
    .catch(() => "pending");

  return {
    state,
    runDialog,
    scheduleDialog,
  };
}

async function openScheduleDialog(page: Page, buildPage: BuildPage) {
  const scheduleButton = page.locator('[data-id="schedule-graph-button"]');

  await dismissSaveToast(page);

  for (let attempt = 0; attempt < 2; attempt++) {
    await expect(scheduleButton).toBeVisible({ timeout: 15000 });
    await expect(scheduleButton).toBeEnabled({ timeout: 15000 });
    await scheduleButton.click();

    const { state, runDialog, scheduleDialog } = await waitForScheduleUi(page);
    if (state !== "pending") {
      return { runDialog, scheduleDialog };
    }

    await page.reload();
    await page.waitForLoadState("domcontentloaded");
    await buildPage.closeTutorial();
    await expect(page.locator(".react-flow")).toBeVisible({ timeout: 15000 });
    await dismissSaveToast(page);
  }

  throw new Error("Schedule UI did not open from the builder");
}

async function startBuilderRun(page: Page, buildPage: BuildPage) {
  await buildPage.clickRunButton();

  const runDialog = page.locator('[data-id="run-input-dialog-content"]');
  if (await runDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.locator('[data-id="run-input-manual-run-button"]').click();
  }
}

async function getBuilderExecutionState(page: Page) {
  const stopButton = page.locator('[data-id="stop-graph-button"]');
  if (await stopButton.isVisible().catch(() => false)) {
    return "running";
  }

  const runButton = page.locator('[data-id="run-graph-button"]');
  if (await runButton.isVisible().catch(() => false)) {
    return "idle";
  }

  return "unknown";
}

async function openSavedAgentInLibrary(page: Page, agentName: string) {
  await page.goto("/library");
  await navigateToAgentByName(page, agentName);
  await waitForAgentPageLoad(page);
}

async function getVisibleExportControl(page: Page) {
  const directExportButton = page.getByRole("button", {
    name: "Export agent to file",
  });
  if (await directExportButton.isVisible().catch(() => false)) {
    return "direct";
  }

  const moreActionsButtons = page.getByRole("button", { name: "More actions" });
  const moreActionsCount = await moreActionsButtons.count();
  for (let index = 0; index < moreActionsCount; index++) {
    if (await moreActionsButtons.nth(index).isVisible().catch(() => false)) {
      return `menu:${index}`;
    }
  }

  return "pending";
}

async function waitForExportControl(page: Page) {
  for (let attempt = 0; attempt < 2; attempt++) {
    let exportControl = "pending";

    await expect
      .poll(
        async () => {
          exportControl = await getVisibleExportControl(page);
          return exportControl;
        },
        { timeout: 15000 },
      )
      .not.toBe("pending")
      .catch(() => {
        exportControl = "pending";
      });

    if (exportControl !== "pending") {
      return exportControl;
    }

    await page.reload();
    await waitForAgentPageLoad(page);
  }

  throw new Error("Export controls did not appear on the agent page");
}

async function clickExportAgent(page: Page) {
  const exportControl = await waitForExportControl(page);
  if (exportControl === "direct") {
    await page
      .getByRole("button", { name: "Export agent to file" })
      .click({ timeout: 15000 });
    return;
  }

  const moreActionsIndex = Number(exportControl.replace("menu:", ""));
  await page
    .getByRole("button", { name: "More actions" })
    .nth(moreActionsIndex)
    .click();

  const dropdownExportButton = page.getByRole("menuitem", {
    name: "Export agent to file",
  });
  await dropdownExportButton.waitFor({ state: "visible", timeout: 15000 });
  await dropdownExportButton.click();
}

async function configureSchedule(page: Page) {
  const hourSelect = page.locator("#time-hour");
  await expect(hourSelect).toBeVisible({ timeout: 15000 });

  const currentHourText = (await hourSelect.textContent()) ?? "";
  const currentHourMatch = currentHourText.match(/\b(1[0-2]|[1-9])\b/);
  const currentHour = currentHourMatch?.[0] ?? "9";
  const nextHour = currentHour === "10" ? "11" : "10";

  await hourSelect.click();

  const nextHourOption = page.getByRole("option", {
    name: nextHour,
    exact: true,
  });
  await nextHourOption.waitFor({ state: "visible", timeout: 15000 });
  await nextHourOption.click();

  await expect(hourSelect).toContainText(nextHour);
}

async function waitForScheduleCreation(page: Page, scheduleDialog: Locator) {
  const successToast = page.getByText("Schedule created");
  const invalidScheduleToast = page.getByText("Invalid schedule");
  const failedScheduleToast = page.getByText("Failed to create schedule");

  await expect
    .poll(
      async () => {
        if (await successToast.isVisible().catch(() => false)) {
          return "success";
        }

        if (!(await scheduleDialog.isVisible().catch(() => false))) {
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
      { timeout: 30000 },
    )
    .toBe("success");
}

test("builder happy path: user can complete or skip the builder tutorial successfully", async ({
  page,
}) => {
  test.setTimeout(90000);

  await openBuilder(page);

  await expect(
    page.getByRole("button", { name: "Skip Tutorial", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".react-flow")).toBeVisible();
});

test("builder happy path: user can create a simple agent in builder with core blocks", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = await openBuilder(page);
  await addSimpleAgentBlocks(buildPage);

  await expect(buildPage.getNodeLocator()).toHaveCount(2);
});

test("builder happy path: user can save the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Save Agent");

  await expect(page).toHaveURL(/flowID=/);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});

test("builder happy path: user can run the saved agent from builder", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Run Agent");

  await startBuilderRun(page, buildPage);
  await expect(
    page.locator('[data-id="stop-graph-button"], [data-id="run-graph-button"]'),
  ).toBeVisible({ timeout: 15000 });
});

test("builder happy path: user can see the run result or execution status", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage } = await createAndSaveAgent(page, "Smoke Status Agent");

  await startBuilderRun(page, buildPage);
  await expect
    .poll(() => getBuilderExecutionState(page), {
      timeout: 15000,
    })
    .not.toBe("unknown");
});

test("builder happy path: user can schedule the saved agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { buildPage, agentName } = await createAndSaveAgent(
    page,
    "Smoke Schedule Agent",
  );

  const { runDialog, scheduleDialog } = await openScheduleDialog(
    page,
    buildPage,
  );

  if (
    (await runDialog.isVisible({ timeout: 1000 }).catch(() => false)) &&
    !(await scheduleDialog.isVisible({ timeout: 1000 }).catch(() => false))
  ) {
    await page.locator('[data-id="run-input-schedule-button"]').click();
  }

  await expect(scheduleDialog).toBeVisible({ timeout: 15000 });
  await page.locator("#schedule-name").fill(`Daily ${agentName}`);
  await configureSchedule(page);

  await page.getByRole("button", { name: "Done" }).click();
  await waitForScheduleCreation(page, scheduleDialog);
  await expect(scheduleDialog).toBeHidden({ timeout: 15000 });
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});

test("builder happy path: user can export the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const { agentName } = await createAndSaveAgent(page, "Smoke Export Agent");

  await openSavedAgentInLibrary(page, agentName);

  const downloadPromise = page
    .waitForEvent("download", { timeout: 15000 })
    .catch(() => null);

  await clickExportAgent(page);

  const download = await downloadPromise;
  if (download) {
    expect(download.suggestedFilename()).toMatch(/\.json$/i);
  }

  await expect(page.getByText("Agent exported")).toBeVisible({
    timeout: 15000,
  });
});
