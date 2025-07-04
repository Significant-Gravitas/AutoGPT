/* eslint-disable react-hooks/rules-of-hooks */
import { test as base } from "@playwright/test";
import fs from "fs";
import path from "path";
import { LoginPage } from "../pages/login.page";
import { TestUser } from "../utils/auth";

// Extend both worker state and test-specific fixtures
type WorkerFixtures = {
  workerAuth: TestUser;
};

type TestFixtures = {
  testUser: TestUser;
  loginPage: LoginPage;
};

export const test = base.extend<TestFixtures, WorkerFixtures>({
  // Define the worker-level fixture that loads pre-created worker-specific auth
  workerAuth: [
    async ({}, use, workerInfo) => {
      const workerId = workerInfo.workerIndex;
      const fileName = path.resolve(
        process.cwd(),
        `.auth/worker-${workerId}.json`,
      );

      if (!fs.existsSync(fileName)) {
        throw new Error(
          `Test user not found for worker ${workerId}. Run global setup first.`,
        );
      }

      const auth: TestUser = JSON.parse(fs.readFileSync(fileName, "utf-8"));
      await use(auth);

      // Cleanup code is commented out to preserve test users during development
      /*
      if (workerInfo.project.metadata.teardown) {
        if (fs.existsSync(fileName)) {
          fs.unlinkSync(fileName);
        }
      }
      */
    },
    { scope: "worker" },
  ],

  // Define the test-level fixture that provides access to the worker auth
  testUser: async ({ workerAuth }, use) => {
    await use(workerAuth);
  },

  // Login page fixture
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
});

export { expect } from "@playwright/test";
