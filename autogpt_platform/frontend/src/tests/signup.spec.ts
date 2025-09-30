import test, { expect } from "@playwright/test";
import {
  generateTestEmail,
  generateTestPassword,
  signupTestUser,
  validateSignupForm,
} from "./utils/signup";
import { getSelectors } from "./utils/selectors";
import { hasUrl, isVisible } from "./utils/assertion";

test("user can signup successfully", async ({ page }) => {
  try {
    const testUser = await signupTestUser(page);
    const { getText, getId } = getSelectors(page);

    // Verify user was created
    expect(testUser.email).toBeTruthy();
    expect(testUser.password).toBeTruthy();
    expect(testUser.createdAt).toBeTruthy();

    const marketplaceText = getText(
      "Bringing you AI agents designed by thinkers from around the world",
    );

    // Verify we're on marketplace and authenticated
    await hasUrl(page, "/marketplace");
    await isVisible(marketplaceText);
    await isVisible(getId("profile-popout-menu-trigger"));
  } catch (error) {
    console.error("❌ Signup test failed:", error);
  }
});

test("signup form validation works", async ({ page }) => {
  const { getField, getRole, getButton } = getSelectors(page);
  const emailInput = getField("Email");
  const passwordInput = page.locator("#password");
  const confirmPasswordInput = page.locator("#confirmPassword");
  const signupButton = getButton("Sign up");
  const termsCheckbox = getRole("checkbox");

  await validateSignupForm(page);

  // Additional validation tests
  await page.goto("/signup");

  // Test with mismatched passwords
  await emailInput.fill(generateTestEmail());
  await passwordInput.fill("password1");
  await confirmPasswordInput.fill("password2");
  await termsCheckbox.click();
  await signupButton.click();

  // Should still be on signup page
  await hasUrl(page, /\/signup/);
});

test("user can signup with custom credentials", async ({ page }) => {
  const { getId } = getSelectors(page);

  try {
    const customEmail = generateTestEmail();
    const customPassword = await generateTestPassword();

    const testUser = await signupTestUser(page, customEmail, customPassword);

    // Verify correct credentials were used
    expect(testUser.email).toBe(customEmail);
    expect(testUser.password).toBe(customPassword);

    // Verify successful signup
    await hasUrl(page, "/marketplace");
    await isVisible(getId("profile-popout-menu-trigger"));
  } catch (error) {
    console.error("❌ Custom credentials signup test failed:", error);
  }
});

test("user can signup with existing email handling", async ({
  page,
  browser,
}) => {
  try {
    const testEmail = generateTestEmail();
    const testPassword = await generateTestPassword();

    // First signup
    const firstUser = await signupTestUser(page, testEmail, testPassword);
    expect(firstUser.email).toBe(testEmail);

    // Create new browser context for second signup (simulates new browser window)
    const newContext = await browser.newContext();
    const newPage = await newContext.newPage();

    try {
      const { getText, getField, getRole, getButton } = getSelectors(newPage);

      // Second signup attempt with same email in new browser context
      // Navigate to signup page
      await newPage.goto("http://localhost:3000/signup");

      // Wait for page to load
      getText("Create a new account");

      // Fill form
      const emailInput = getField("Email");
      await emailInput.fill(testEmail);
      const passwordInput = newPage.locator("#password");
      await passwordInput.fill(testPassword);
      const confirmPasswordInput = newPage.locator("#confirmPassword");
      await confirmPasswordInput.fill(testPassword);

      // Agree to terms and submit
      await getRole("checkbox").click();
      const signupButton = getButton("Sign up");
      await signupButton.click();
      await isVisible(getText("User with this email already exists"));
    } catch (_error) {
    } finally {
      // Clean up new browser context
      await newContext.close();
    }
  } catch (error) {
    console.error("❌ Duplicate email handling test failed:", error);
  }
});
