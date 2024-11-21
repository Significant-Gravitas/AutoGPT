/* eslint-disable react-hooks/rules-of-hooks */
import { test as base } from "@playwright/test";
import { LoginPage } from "../pages/login.page";

export const loginPageFixture = base.extend<{ loginPage: LoginPage }>({
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
});

// Export just the fixture function
export const createLoginPageFixture = async ({ page }, use) => {
  await use(new LoginPage(page));
};
