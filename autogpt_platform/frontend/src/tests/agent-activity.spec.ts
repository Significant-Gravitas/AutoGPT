import test, { expect } from "@playwright/test";
import { hasTextContent, hasUrl, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";
import { BuildPage } from "./pages/build.page";
import * as LibraryPage from "./pages/library.page";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  const buildPage = new BuildPage(page);
  const testUser = await getTestUser();

  const { getId, getText } = getSelectors(page);

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
  await LibraryPage.clickFirstAgent(page);
  await LibraryPage.waitForAgentPageLoad(page);
  await isVisible(getText("Test Agent"), 8000);
});

test("shows badge with count when agent is running", async ({ page }) => {
  const { getId } = getSelectors(page);

  await LibraryPage.clickRunButton(page);

  const badge = getId("agent-activity-badge");
  await isVisible(badge);

  await expect(async () => {
    try {
      await hasTextContent(badge, "1");
    } catch {
      await hasTextContent(badge, "2");
    }
  }).toPass();
});

test("displays the runs on the activity dropdown", async ({ page }) => {
  const { getId } = getSelectors(page);

  const activityBtn = getId("agent-activity-button");
  await isVisible(activityBtn);

  await LibraryPage.clickRunButton(page);

  await activityBtn.click();

  const dropdown = getId("agent-activity-dropdown");
  await isVisible(dropdown);

  await hasTextContent(dropdown, "Test Agent");

  // Check for either running or queued state
  const runningText = "Started just now, a few seconds running";
  const queuedText =
    "Test AgentStarted just now, a few seconds queuedTest AgentStarted just now, a few seconds queued";

  await expect(async () => {
    try {
      await hasTextContent(dropdown, runningText);
    } catch {
      await hasTextContent(dropdown, queuedText);
    }
  }).toPass();
});
