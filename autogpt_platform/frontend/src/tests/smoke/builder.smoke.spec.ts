import { expect, test } from "../coverage-fixture";
import { SMOKE_AUTH_STATES } from "../credentials/accounts";
import { BuildPage } from "../pages/build.page";

test.use({ storageState: SMOKE_AUTH_STATES.builder });

test("@smoke builder flow: user can create, run, and schedule an agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const agentName = `Smoke Builder Agent ${Date.now()}`;

  await page.goto("/build");
  await page.waitForLoadState("domcontentloaded");
  await buildPage.closeTutorial();

  await buildPage.addBlockByClick("Store Value");
  await buildPage.waitForNodeOnCanvas(1);

  await buildPage.saveAgent(agentName, "PR smoke builder coverage");
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  await buildPage.clickRunButton();

  const runDialog = page.locator('[data-id="run-input-dialog-content"]');
  if (await runDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.locator('[data-id="run-input-manual-run-button"]').click();
  }

  await expect(
    page.locator('[data-id="stop-graph-button"], [data-id="run-graph-button"]'),
  ).toBeVisible({ timeout: 15000 });

  await page.locator('[data-id="schedule-graph-button"]').click();

  if (await runDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.locator('[data-id="run-input-schedule-button"]').click();
  }

  await expect(
    page.getByRole("dialog", { name: "Schedule Graph" }),
  ).toBeVisible();
  await page.locator("#schedule-name").fill(`Daily ${agentName}`);
  await page.getByRole("button", { name: "Done" }).click();

  await expect(page.getByText("Schedule created")).toBeVisible({
    timeout: 15000,
  });
});
