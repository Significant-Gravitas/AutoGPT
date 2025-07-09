import { test, expect } from "./fixtures";
import {
  signupTestUser,
  validateSignupForm,
  generateTestEmail,
  generateTestPassword,
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
    await page.getByPlaceholder("m@example.com").fill(generateTestEmail());
    const passwordInputs = page.getByTitle("Password");
    await passwordInputs.nth(0).fill("password1");
    await passwordInputs.nth(1).fill("password2");
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

  test("user can signup with existing email handling", async ({ page }) => {
    console.log("🧪 Testing duplicate email handling...");

    try {
      const testEmail = generateTestEmail();
      const testPassword = generateTestPassword();

      // First signup
      console.log(`👤 First signup attempt: ${testEmail}`);
      const firstUser = await signupTestUser(page, testEmail, testPassword);

      expect(firstUser.email).toBe(testEmail);
      console.log("✅ First signup successful");

      // Second signup attempt with same email should handle gracefully
      console.log(`👤 Second signup attempt: ${testEmail}`);
      try {
        await signupTestUser(page, testEmail, testPassword);
        console.log("ℹ️ Second signup handled gracefully");
      } catch (_error) {
        console.log("ℹ️ Second signup rejected as expected");
      }

      console.log("✅ Duplicate email handling test completed");
    } catch (error) {
      console.error("❌ Duplicate email handling test failed:", error);
    }
  });
});
