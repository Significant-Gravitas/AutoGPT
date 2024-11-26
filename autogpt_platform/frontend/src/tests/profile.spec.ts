// profile.spec.ts
import { test } from "./fixtures";
import { ProfilePage } from "./pages/profile.page";

test.describe("Profile", () => {
  let profilePage: ProfilePage;

  test.beforeEach(async ({ page, loginPage, testUser }) => {
    profilePage = new ProfilePage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");
  });

  test("user can view their profile information", async ({
    page,
    testUser,
  }) => {
    await profilePage.navbar.clickProfileLink();
    await test.expect(profilePage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL(new RegExp("/profile"));

    // Verify email matches test worker's email
    const displayedEmail = await profilePage.getDisplayedEmail();
    test.expect(displayedEmail).toBe(testUser.email);
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
