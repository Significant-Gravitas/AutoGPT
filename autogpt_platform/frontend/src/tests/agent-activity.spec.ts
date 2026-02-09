import test, { expect } from "@playwright/test";
import { BuildPage } from "./pages/build.page";
import * as LibraryPage from "./pages/library.page";
import { LoginPage } from "./pages/login.page";
import { hasTextContent, hasUrl, isVisible } from "./utils/assertion";
import { getTestUser } from "./utils/auth";
import { getSelectors } from "./utils/selectors";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  const buildPage = new BuildPage(page);
  const testUser = await getTestUser();

  const { getId } = getSelectors(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  await page.goto("/build");
  await buildPage.closeTutorial();
  await buildPage.openBlocksPanel();

  const [dictionaryBlock] = await buildPage.getFilteredBlocksFromAPI(
    (block) => block.name === "AddToDictionaryBlock",
  );

  const blockCard = getId(`block-name-${dictionaryBlock.id}`);
  await blockCard.click();
  const blockInEditor = getId(dictionaryBlock.id).first();
  expect(blockInEditor).toBeAttached();

  await buildPage.saveAgent("Test Agent", "Test Description");
  await test
    .expect(page)
    .toHaveURL(({ searchParams }) => !!searchParams.get("flowID"));

  // Wait for save to complete
  await page.waitForTimeout(1000);

  await page.goto("/library");
  // Navigate to the specific agent we just created, not just the first one
  await LibraryPage.navigateToAgentByName(page, "Test Agent");
  await LibraryPage.waitForAgentPageLoad(page);
});

test("shows badge with count when agent is running", async ({ page }) => {
  const { getId } = getSelectors(page);

  // Start the agent run
  await LibraryPage.clickRunButton(page);

  // Wait for the badge to appear and check it has a valid count
  const badge = getId("agent-activity-badge");
  await isVisible(badge);

  // Check that badge shows a positive number (more flexible than exact count)
  await expect(async () => {
    const badgeText = await badge.textContent();
    const count = parseInt(badgeText || "0");

    if (count < 1) {
      throw new Error(`Expected badge count >= 1, got: ${badgeText}`);
    }
  }).toPass({ timeout: 10000 });
});

test("displays the runs on the activity dropdown", async ({ page }) => {
  const { getId } = getSelectors(page);

  const activityBtn = getId("agent-activity-button");
  await isVisible(activityBtn);

  // Start the agent run
  await LibraryPage.clickRunButton(page);

  // Wait for the activity badge to appear (indicating execution started)
  const badge = getId("agent-activity-badge");
  await isVisible(badge);

  // Click to open the dropdown
  await activityBtn.click();

  const dropdown = getId("agent-activity-dropdown");
  await isVisible(dropdown);

  // Check that the agent name appears in the dropdown
  await hasTextContent(dropdown, "Test Agent");

  // Check for execution status - be more flexible with text matching
  await expect(async () => {
    const dropdownText = await dropdown.textContent();
    const hasAgentName = dropdownText?.includes("Test Agent");
    const hasExecutionStatus =
      dropdownText?.includes("queued") ||
      dropdownText?.includes("running") ||
      dropdownText?.includes("Started");

    if (!hasAgentName || !hasExecutionStatus) {
      throw new Error(
        `Expected agent name and execution status, got: ${dropdownText}`,
      );
    }
  }).toPass({ timeout: 8000 });
});
