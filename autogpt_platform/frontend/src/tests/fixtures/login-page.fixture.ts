/* eslint-disable react-hooks/rules-of-hooks */
import { test as base } from "@playwright/test";
import { LoginPage } from "../pages/login.page";
import { Page } from "@playwright/test";
export const loginPageFixture = base.extend<{ loginPage: LoginPage }>({
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
});

// Export just the fixture function
export const createLoginPageFixture = async (
  { page }: { page: Page },
  use: (loginPage: LoginPage) => Promise<void>,
) => {
  await use(new LoginPage(page));
};
