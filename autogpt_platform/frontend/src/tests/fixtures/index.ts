import { test as base } from "@playwright/test";
import { createTestUserFixture } from "./test-user.fixture";
import { createLoginPageFixture } from "./login-page.fixture";
import type { TestUser } from "./test-user.fixture";
import { LoginPage } from "../pages/login.page";

type Fixtures = {
  testUser: TestUser;
  loginPage: LoginPage;
};

// Combine fixtures
export const test = base.extend<Fixtures>({
  testUser: createTestUserFixture,
  loginPage: createLoginPageFixture,
});

export { expect } from "@playwright/test";
