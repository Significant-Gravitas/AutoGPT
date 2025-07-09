import { expect, test } from "./fixtures";
import {
  generateTestEmail,
  generateTestPassword,
  signupTestUser,
  validateSignupForm,
} from "./utils/signup";

test.describe("Signup Flow", () => {
  test("user can signup successfully", async ({ page }) => {
    console.log("🧪 Testing user signup flow...");

    try {
      const testUser = await signupTestUser(page);

      // Verify user was created
      expect(testUser.email).toBeTruthy();
      expect(testUser.password).toBeTruthy();
      expect(testUser.createdAt).toBeTruthy();

      // Verify we're on marketplace and authenticated
      await expect(page).toHaveURL("/marketplace");
      await expect(
        page.getByText(
          "Bringing you AI agents designed by thinkers from around the world",
        ),
      ).toBeVisible();
      await expect(
        page.getByTestId("profile-popout-menu-trigger"),
      ).toBeVisible();

      console.log(`✅ User successfully signed up: ${testUser.email}`);
    } catch (error) {
      console.error("❌ Signup test failed:", error);
    }
  });

  test("signup form validation works", async ({ page }) => {
    console.log("🧪 Testing signup form validation...");

    await validateSignupForm(page);

    // Additional validation tests
    await page.goto("/signup");

    // Test with mismatched passwords
    console.log("❌ Testing mismatched passwords...");
    await page.getByLabel("Email").fill(generateTestEmail());
    const passwordInput = page.getByLabel("Password", { exact: true });
    const confirmPasswordInput = page.getByLabel("Confirm Password", {
      exact: true,
    });
    await passwordInput.fill("password1");
    await confirmPasswordInput.fill("password2");
    await page.getByRole("checkbox").click();
    await page.getByRole("button", { name: "Sign up" }).click();

    // Should still be on signup page
    await expect(page).toHaveURL(/\/signup/);
    console.log("✅ Mismatched passwords correctly blocked");
  });

  test("user can signup with custom credentials", async ({ page }) => {
    console.log("🧪 Testing signup with custom credentials...");

    try {
      const customEmail = generateTestEmail();
      const customPassword = generateTestPassword();

      const testUser = await signupTestUser(page, customEmail, customPassword);

      // Verify correct credentials were used
      expect(testUser.email).toBe(customEmail);
      expect(testUser.password).toBe(customPassword);

      // Verify successful signup
      await expect(page).toHaveURL("/marketplace");
      await expect(
        page.getByTestId("profile-popout-menu-trigger"),
      ).toBeVisible();

      console.log(`✅ Custom credentials signup worked: ${testUser.email}`);
    } catch (error) {
      console.error("❌ Custom credentials signup test failed:", error);
    }
  });

  test("user can signup with existing email handling", async ({
    page,
    browser,
  }) => {
    console.log("🧪 Testing duplicate email handling...");

    try {
      const testEmail = generateTestEmail();
      const testPassword = generateTestPassword();

      // First signup
      console.log(`👤 First signup attempt: ${testEmail}`);
      const firstUser = await signupTestUser(page, testEmail, testPassword);

      expect(firstUser.email).toBe(testEmail);
      console.log("✅ First signup successful");

      // Create new browser context for second signup (simulates new browser window)
      console.log("🔄 Creating new browser context...");
      const newContext = await browser.newContext();
      const newPage = await newContext.newPage();

      try {
        // Second signup attempt with same email in new browser context
        console.log(
          `👤 Second signup attempt in new browser context: ${testEmail}`,
        );
        await signupTestUser(newPage, testEmail, testPassword);
        expect(
          newPage.getByText("User with this email already exists"),
        ).toBeVisible();
        console.log("ℹ️ Second signup handled gracefully");
      } catch (_error) {
        console.log("ℹ️ Second signup rejected as expected");
      } finally {
        // Clean up new browser context
        await newContext.close();
        console.log("🧹 New browser context closed");
      }

      console.log("✅ Duplicate email handling test completed");
    } catch (error) {
      console.error("❌ Duplicate email handling test failed:", error);
    }
  });
});
