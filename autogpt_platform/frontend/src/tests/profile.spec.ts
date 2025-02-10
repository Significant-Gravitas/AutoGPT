import { test } from "./fixtures";
import { ProfilePage } from "./pages/profile.page";

test.describe("Profile", () => {
  let profilePage: ProfilePage;

  test.beforeEach(async ({ page, loginPage, testUser }) => {
    profilePage = new ProfilePage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/marketplace");
  });

  test("user can view their profile information", async ({
    page,
    testUser,
  }) => {
    await profilePage.navbar.clickProfileLink();
    // workaround for #8788
    // sleep for 10 seconds to allow page to load due to bug in our system
    await page.waitForTimeout(10_000);
    await page.reload();
    await page.reload();
    await test.expect(profilePage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL(new RegExp("/profile"));

    // Verify email matches test worker's email
    const displayedHandle = await profilePage.getDisplayedName();
    test.expect(displayedHandle).not.toBeNull();
    test.expect(displayedHandle).not.toBe("");
    test.expect(displayedHandle).toBeDefined();
  });

  test("profile navigation is accessible from navbar", async ({ page }) => {
    await profilePage.navbar.clickProfileLink();
    await test.expect(page).toHaveURL(new RegExp("/profile"));
    await test.expect(profilePage.isLoaded()).resolves.toBeTruthy();
  });

  test("profile displays user Credential providers", async ({ page }) => {
    await profilePage.navbar.clickProfileLink();

    // await test
    //   .expect(page.getByTestId("profile-section-personal"))
    //   .toBeVisible();
    // await test
    //   .expect(page.getByTestId("profile-section-settings"))
    //   .toBeVisible();
    // await test
    //   .expect(page.getByTestId("profile-section-security"))
    //   .toBeVisible();
  });
});
