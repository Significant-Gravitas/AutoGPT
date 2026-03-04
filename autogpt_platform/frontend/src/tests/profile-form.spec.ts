import test, { expect } from "@playwright/test";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { ProfileFormPage } from "./pages/profile-form.page";
import { hasUrl } from "./utils/assertion";

test.describe("Profile Form", () => {
  let profileFormPage: ProfileFormPage;

  test.beforeEach(async ({ page }) => {
    profileFormPage = new ProfileFormPage(page);

    const loginPage = new LoginPage(page);
    await loginPage.goto();
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
    await hasUrl(page, "/marketplace");
  });

  test("redirects to login when user is not authenticated", async ({
    browser,
  }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    try {
      await page.goto("/profile");
      await hasUrl(page, "/login?next=%2Fprofile");
    } finally {
      await page.close();
      await context.close();
    }
  });

  test("can save profile changes successfully", async ({ page }) => {
    await profileFormPage.navbar.clickProfileLink();

    await expect(profileFormPage.isLoaded()).resolves.toBeTruthy();
    await hasUrl(page, new RegExp("/profile"));

    const suffix = Date.now().toString().slice(-6);
    const newDisplayName = `E2E Name ${suffix}`;
    const newHandle = `e2euser${suffix}`;
    const newBio = `E2E bio ${suffix}`;
    const newLinks = [
      `https://example.com/${suffix}/1`,
      `https://example.com/${suffix}/2`,
      `https://example.com/${suffix}/3`,
      `https://example.com/${suffix}/4`,
      `https://example.com/${suffix}/5`,
    ];

    await profileFormPage.setDisplayName(newDisplayName);
    await profileFormPage.setHandle(newHandle);
    await profileFormPage.setBio(newBio);
    await profileFormPage.setLinks(newLinks);
    await profileFormPage.saveChanges();

    expect(await profileFormPage.getDisplayName()).toBe(newDisplayName);
    expect(await profileFormPage.getHandle()).toBe(newHandle);
    expect(await profileFormPage.getBio()).toBe(newBio);
    for (let i = 1; i <= 5; i++) {
      expect(await profileFormPage.getLink(i)).toBe(newLinks[i - 1]);
    }

    await page.reload();
    await expect(profileFormPage.isLoaded()).resolves.toBeTruthy();

    expect(await profileFormPage.getDisplayName()).toBe(newDisplayName);
    expect(await profileFormPage.getHandle()).toBe(newHandle);
    expect(await profileFormPage.getBio()).toBe(newBio);
    for (let i = 1; i <= 5; i++) {
      expect(await profileFormPage.getLink(i)).toBe(newLinks[i - 1]);
    }
  });

  // Currently we are not using hook form inside the profile form, so cancel button is not working as expected, once that's fixed, we can unskip this test
  test.skip("can cancel profile changes", async ({ page }) => {
    await profileFormPage.navbar.clickProfileLink();

    await expect(profileFormPage.isLoaded()).resolves.toBeTruthy();
    await hasUrl(page, new RegExp("/profile"));

    const originalDisplayName = await profileFormPage.getDisplayName();
    const originalHandle = await profileFormPage.getHandle();
    const originalBio = await profileFormPage.getBio();
    const originalLinks: string[] = [];
    for (let i = 1; i <= 5; i++) {
      originalLinks.push(await profileFormPage.getLink(i));
    }

    const suffix = `${Date.now().toString().slice(-6)}_cancel`;
    await profileFormPage.setDisplayName(`Tmp Name ${suffix}`);
    await profileFormPage.setHandle(`tmpuser${suffix}`);
    await profileFormPage.setBio(`Tmp bio ${suffix}`);
    for (let i = 1; i <= 5; i++) {
      await profileFormPage.setLink(i, `https://tmp.example/${suffix}/${i}`);
    }

    await profileFormPage.clickCancel();

    expect(await profileFormPage.getDisplayName()).toBe(originalDisplayName);
    expect(await profileFormPage.getHandle()).toBe(originalHandle);
    expect(await profileFormPage.getBio()).toBe(originalBio);
    for (let i = 1; i <= 5; i++) {
      expect(await profileFormPage.getLink(i)).toBe(originalLinks[i - 1]);
    }
  });
});
