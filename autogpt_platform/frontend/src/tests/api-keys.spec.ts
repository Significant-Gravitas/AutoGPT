import { expect, test } from "@playwright/test";
import { LoginPage } from "./pages/login.page";
import { TEST_CREDENTIALS } from "./credentials";
import { hasUrl } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.describe("API Keys Page", () => {
  test.beforeEach(async ({ page }) => {
    const loginPage = new LoginPage(page);
    await page.goto("/login");
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");
  });

  test("should redirect to login page when user is not authenticated", async ({
    browser,
  }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    try {
      await page.goto("/profile/api-keys");
      await hasUrl(page, "/login?next=%2Fprofile%2Fapi-keys");
    } finally {
      await page.close();
      await context.close();
    }
  });

  test("should create a new API key successfully", async ({ page }) => {
    const { getButton, getField } = getSelectors(page);
    await page.goto("/profile/api-keys");
    await getButton("Create Key").click();

    await getField("Name").fill("Test Key");
    await getButton("Create").click();

    await expect(
      page.getByText("AutoGPT Platform API Key Created"),
    ).toBeVisible();
    await getButton("Close").first().click();

    await expect(page.getByText("Test Key").first()).toBeVisible();
  });

  test("should revoke an existing API key", async ({ page }) => {
    const { getRole, getId } = getSelectors(page);
    await page.goto("/profile/api-keys");

    const apiKeyRow = getId("api-key-row").first();
    const apiKeyContent = await apiKeyRow
      .getByTestId("api-key-id")
      .textContent();
    const apiKeyActions = apiKeyRow.getByTestId("api-key-actions").first();

    await apiKeyActions.click();
    await getRole("menuitem", "Revoke").click();
    await expect(
      page.getByText("AutoGPT Platform API key revoked successfully"),
    ).toBeVisible();

    await expect(page.getByText(apiKeyContent!)).not.toBeVisible();
  });
});
