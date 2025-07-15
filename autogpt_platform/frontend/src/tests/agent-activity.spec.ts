import test from "@playwright/test";
import {
  navigateToLibrary,
  clickFirstAgent,
  waitForAgentPageLoad,
  runAgent,
} from "./pages/library.page";
import { hasTextContent, hasUrl, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  const testUser = await getTestUser();

  const { getText, getId } = getSelectors(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");
  await navigateToLibrary(page);
  await hasUrl(page, new RegExp("/library"));

  await clickFirstAgent(page);
  await waitForAgentPageLoad(page);

  // wait for page to load
  const activityBtn = getId("agent-activity-button");
  await isVisible(activityBtn);
  await isVisible(getText("Run actions"));
});

test("shows badge with count when agent is running", async ({ page }) => {
  const { getId } = getSelectors(page);

  await runAgent(page);

  const badge = getId("agent-activity-badge");
  await isVisible(badge);

  await hasTextContent(badge, "1");
});

test("shows hover hint when agent is running", async ({ page }) => {
  const { getId, getRole } = getSelectors(page);

  await runAgent(page);

  const activityBtn = getId("agent-activity-button");
  await isVisible(activityBtn);

  await activityBtn.hover();

  const hint = getRole("tooltip");
  await isVisible(hint);

  await hasTextContent(hint, "1 running agent");
});
