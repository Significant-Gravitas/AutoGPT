import { test } from "./fixtures";
import { LibraryPage } from "./pages/library.page";

test.describe("Agent Notifications", () => {
  let libraryPage: LibraryPage;

  test.beforeEach(async ({ page, loginPage, testUser }) => {
    libraryPage = new LibraryPage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/marketplace");
  });

  test("notification badge appears when agent is running", async ({ page }) => {
    // Navigate to library
    await libraryPage.navigateToLibrary();
    await test.expect(page).toHaveURL(new RegExp("/library"));

    // Click on first available agent
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    // Verify we're on agent page
    await test.expect(libraryPage.isLoaded()).resolves.toBeTruthy();

    // Initially, no notification badge should be visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationBadgeVisible())
      .resolves.toBeFalsy();

    // Run the agent
    await libraryPage.runAgent();

    // Wait for run to start and notification badge to appear
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Check that notification badge appears
    await test
      .expect(libraryPage.agentNotifications.isNotificationBadgeVisible())
      .resolves.toBeTruthy();

    // Check that notification count is at least 1
    const notificationCount =
      await libraryPage.agentNotifications.getNotificationCount();
    test.expect(parseInt(notificationCount)).toBeGreaterThan(0);
  });

  test("notification dropdown shows running agent with correct status", async ({
    page: _page,
  }) => {
    // Navigate to library and run an agent
    await libraryPage.navigateToLibrary();
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    const agentName = await libraryPage.getAgentName();

    // Run the agent
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Click on notification button to open dropdown
    await libraryPage.agentNotifications.clickNotificationButton();

    // Verify dropdown is visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationDropdownVisible())
      .resolves.toBeTruthy();

    // Check that running agent appears in dropdown
    await test
      .expect(
        libraryPage.agentNotifications.hasNotificationWithStatus("running"),
      )
      .resolves.toBeTruthy();

    // Check that the agent name appears in notifications
    const notification =
      await libraryPage.agentNotifications.getNotificationByAgentName(
        agentName,
      );
    test.expect(notification).not.toBeNull();
    test.expect(notification?.status).toBe("running");
  });

  test("notification dropdown shows completed agent after run finishes", async ({
    page: _page,
  }, testInfo) => {
    // Increase timeout for this test since we need to wait for completion
    await test.setTimeout(testInfo.timeout * 3);

    // Navigate to library and run an agent
    await libraryPage.navigateToLibrary();
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    const agentName = await libraryPage.getAgentName();

    // Run the agent
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Wait for agent to complete (with longer timeout)
    await libraryPage.waitForRunToComplete(60000);
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Click on notification button to open dropdown
    await libraryPage.agentNotifications.clickNotificationButton();

    // Verify dropdown is visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationDropdownVisible())
      .resolves.toBeTruthy();

    // Check that completed agent appears in dropdown
    const notification =
      await libraryPage.agentNotifications.getNotificationByAgentName(
        agentName,
      );
    test.expect(notification).not.toBeNull();
    test.expect(notification?.status).toMatch(/completed|failed|terminated/);
  });

  test("notification dropdown shows correct time information", async ({
    page: _page,
  }) => {
    // Navigate to library and run an agent
    await libraryPage.navigateToLibrary();
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    const agentName = await libraryPage.getAgentName();

    // Run the agent
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Click on notification button to open dropdown
    await libraryPage.agentNotifications.clickNotificationButton();

    // Get notification for this agent
    const notification =
      await libraryPage.agentNotifications.getNotificationByAgentName(
        agentName,
      );
    test.expect(notification).not.toBeNull();

    // Check that time information is present and contains expected text
    test.expect(notification?.time).toContain("Started");
    test.expect(notification?.time).toMatch(/Started.*ago.*seconds/);
  });

  test("notification dropdown shows multiple agents when multiple are running", async ({
    page: _page,
  }) => {
    // Navigate to library
    await libraryPage.navigateToLibrary();

    // Run first agent
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();
    const firstAgentName = await libraryPage.getAgentName();
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Go back to library and run another agent (if available)
    await libraryPage.navigateToLibrary();
    const agentCards = await libraryPage.agentCards.count();

    if (agentCards > 1) {
      await libraryPage.agentCards.nth(1).click();
      await libraryPage.waitForAgentPageLoad();
      const secondAgentName = await libraryPage.getAgentName();
      await libraryPage.runAgent();
      await libraryPage.agentNotifications.waitForNotificationUpdate();

      // Click on notification button to open dropdown
      await libraryPage.agentNotifications.clickNotificationButton();

      // Verify both agents appear in dropdown
      const firstNotification =
        await libraryPage.agentNotifications.getNotificationByAgentName(
          firstAgentName,
        );
      const secondNotification =
        await libraryPage.agentNotifications.getNotificationByAgentName(
          secondAgentName,
        );

      test.expect(firstNotification).not.toBeNull();
      test.expect(secondNotification).not.toBeNull();

      // Check that notification count reflects multiple running agents
      const notificationCount =
        await libraryPage.agentNotifications.getNotificationCount();
      test.expect(parseInt(notificationCount)).toBeGreaterThanOrEqual(2);
    } else {
      // Skip this part if only one agent is available
      console.log("Only one agent available, skipping multiple agent test");
    }
  });

  test("notification dropdown closes when clicking outside", async ({
    page,
  }) => {
    // Navigate to library and run an agent
    await libraryPage.navigateToLibrary();
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    // Run the agent
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Click on notification button to open dropdown
    await libraryPage.agentNotifications.clickNotificationButton();

    // Verify dropdown is visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationDropdownVisible())
      .resolves.toBeTruthy();

    // Click outside the dropdown
    await page.click("body");

    // Verify dropdown is no longer visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationDropdownVisible())
      .resolves.toBeFalsy();
  });

  test("notification badge count updates correctly", async ({
    page: _page,
  }) => {
    // Navigate to library and run an agent
    await libraryPage.navigateToLibrary();
    await libraryPage.clickFirstAgent();
    await libraryPage.waitForAgentPageLoad();

    // Initially, no notification badge should be visible
    await test
      .expect(libraryPage.agentNotifications.isNotificationBadgeVisible())
      .resolves.toBeFalsy();

    // Run the agent
    await libraryPage.runAgent();
    await libraryPage.agentNotifications.waitForNotificationUpdate();

    // Check that notification count is 1
    const notificationCount =
      await libraryPage.agentNotifications.getNotificationCount();
    test.expect(notificationCount).toBe("1");

    // Check that badge is visible and animating
    await test
      .expect(libraryPage.agentNotifications.isNotificationBadgeVisible())
      .resolves.toBeTruthy();
  });
});
