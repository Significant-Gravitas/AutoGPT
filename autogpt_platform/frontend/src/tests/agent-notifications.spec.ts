import { test } from "./fixtures";
import {
  navigateToLibrary,
  clickFirstAgent,
  waitForAgentPageLoad,
  getAgentName,
  runAgent,
  waitForRunToComplete,
  isLoaded,
  getAgentCards,
} from "./pages/library.page";
import {
  clickNotificationButton,
  isNotificationBadgeVisible,
  isNotificationDropdownVisible,
  getNotificationCount,
  waitForNotificationUpdate,
  hasNotificationWithStatus,
  getNotificationByAgentName,
} from "./pages/header.page";

test.beforeEach(async ({ page, loginPage, testUser }) => {
  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await test.expect(page).toHaveURL("/marketplace");
});

test("shows badge with count when agent is running", async ({ page }) => {
  await navigateToLibrary(page);
  await test.expect(page).toHaveURL(new RegExp("/library"));

  await clickFirstAgent(page);
  await waitForAgentPageLoad(page);

  await test.expect(isLoaded(page)).resolves.toBeTruthy();
  await test.expect(isNotificationBadgeVisible(page)).resolves.toBeFalsy();

  await runAgent(page);

  await waitForNotificationUpdate(page);

  await test.expect(isNotificationBadgeVisible(page)).resolves.toBeTruthy();

  // Check that notification count is at least 1
  const notificationCount = await getNotificationCount(page);
  test.expect(parseInt(notificationCount)).toBeGreaterThan(0);
});

test("shows running agent with correct status in dropdown", async ({
  page: _page,
}) => {
  await navigateToLibrary(_page);
  await clickFirstAgent(_page);
  await waitForAgentPageLoad(_page);

  const agentName = await getAgentName(_page);

  // Run the agent
  await runAgent(_page);
  await waitForNotificationUpdate(_page);

  // Click on notification button to open dropdown
  await clickNotificationButton(_page);

  // Verify dropdown is visible
  await test.expect(isNotificationDropdownVisible(_page)).resolves.toBeTruthy();

  // Check that running agent appears in dropdown
  await test
    .expect(hasNotificationWithStatus(_page, "running"))
    .resolves.toBeTruthy();

  // Check that the agent name appears in notifications
  const notification = await getNotificationByAgentName(_page, agentName);
  test.expect(notification).not.toBeNull();
  test.expect(notification?.status).toBe("running");
});

test("shows completed agent after run finishes", async ({
  page: _page,
}, testInfo) => {
  // Increase timeout for this test since we need to wait for completion
  await test.setTimeout(testInfo.timeout * 3);

  // Navigate to library and run an agent
  await navigateToLibrary(_page);
  await clickFirstAgent(_page);
  await waitForAgentPageLoad(_page);

  const agentName = await getAgentName(_page);

  // Run the agent
  await runAgent(_page);
  await waitForNotificationUpdate(_page);

  // Wait for agent to complete (with longer timeout)
  await waitForRunToComplete(_page, 60000);
  await waitForNotificationUpdate(_page);

  // Click on notification button to open dropdown
  await clickNotificationButton(_page);

  // Verify dropdown is visible
  await test.expect(isNotificationDropdownVisible(_page)).resolves.toBeTruthy();

  // Check that completed agent appears in dropdown
  const notification = await getNotificationByAgentName(_page, agentName);
  test.expect(notification).not.toBeNull();
  test.expect(notification?.status).toMatch(/completed|failed|terminated/);
});

test("shows correct time information", async ({ page: _page }) => {
  // Navigate to library and run an agent
  await navigateToLibrary(_page);
  await clickFirstAgent(_page);
  await waitForAgentPageLoad(_page);

  const agentName = await getAgentName(_page);

  // Run the agent
  await runAgent(_page);
  await waitForNotificationUpdate(_page);

  // Click on notification button to open dropdown
  await clickNotificationButton(_page);

  // Get notification for this agent
  const notification = await getNotificationByAgentName(_page, agentName);
  test.expect(notification).not.toBeNull();

  // Check that time information is present and contains expected text
  test.expect(notification?.time).toContain("Started");
  test.expect(notification?.time).toMatch(/Started.*ago.*seconds/);
});

test("shows multiple agents when multiple are running", async ({
  page: _page,
}) => {
  // Navigate to library
  await navigateToLibrary(_page);

  // Run first agent
  await clickFirstAgent(_page);
  await waitForAgentPageLoad(_page);
  const firstAgentName = await getAgentName(_page);
  await runAgent(_page);
  await waitForNotificationUpdate(_page);

  // Go back to library and run another agent (if available)
  await navigateToLibrary(_page);
  const agentCards = await getAgentCards(_page).count();

  if (agentCards > 1) {
    await getAgentCards(_page).nth(1).click();
    await waitForAgentPageLoad(_page);
    const secondAgentName = await getAgentName(_page);
    await runAgent(_page);
    await waitForNotificationUpdate(_page);

    // Click on notification button to open dropdown
    await clickNotificationButton(_page);

    // Verify both agents appear in dropdown
    const firstNotification = await getNotificationByAgentName(
      _page,
      firstAgentName,
    );
    const secondNotification = await getNotificationByAgentName(
      _page,
      secondAgentName,
    );

    test.expect(firstNotification).not.toBeNull();
    test.expect(secondNotification).not.toBeNull();

    // Check that notification count reflects multiple running agents
    const notificationCount = await getNotificationCount(_page);
    test.expect(parseInt(notificationCount)).toBeGreaterThanOrEqual(2);
  } else {
    // Skip this part if only one agent is available
    console.log("Only one agent available, skipping multiple agent test");
  }
});

test("closes when clicking outside", async ({ page }) => {
  // Navigate to library and run an agent
  await navigateToLibrary(page);
  await clickFirstAgent(page);
  await waitForAgentPageLoad(page);

  // Run the agent
  await runAgent(page);
  await waitForNotificationUpdate(page);

  // Click on notification button to open dropdown
  await clickNotificationButton(page);

  // Verify dropdown is visible
  await test.expect(isNotificationDropdownVisible(page)).resolves.toBeTruthy();

  // Click outside the dropdown
  await page.click("body");

  // Verify dropdown is no longer visible
  await test.expect(isNotificationDropdownVisible(page)).resolves.toBeFalsy();
});

test("updates badge count", async ({ page: _page }) => {
  // Navigate to library and run an agent
  await navigateToLibrary(_page);
  await clickFirstAgent(_page);
  await waitForAgentPageLoad(_page);

  // Initially, no notification badge should be visible
  await test.expect(isNotificationBadgeVisible(_page)).resolves.toBeFalsy();

  // Run the agent
  await runAgent(_page);
  await waitForNotificationUpdate(_page);

  // Check that notification count is 1
  const notificationCount = await getNotificationCount(_page);
  test.expect(notificationCount).toBe("1");

  // Check that badge is visible and animating
  await test.expect(isNotificationBadgeVisible(_page)).resolves.toBeTruthy();
});
