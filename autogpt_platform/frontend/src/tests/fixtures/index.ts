/* eslint-disable react-hooks/rules-of-hooks */
import { test as base } from "@playwright/test";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { faker } from "@faker-js/faker";
import fs from "fs";
import path from "path";
import { TestUser } from "./test-user.fixture";
import { LoginPage } from "../pages/login.page";

// Extend both worker state and test-specific fixtures
type WorkerFixtures = {
  workerAuth: TestUser;
};

type TestFixtures = {
  testUser: TestUser;
  loginPage: LoginPage;
};

let supabase: SupabaseClient;

function getSupabaseAdmin() {
  if (!supabase) {
    supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false,
        },
      },
    );
  }
  return supabase;
}

export const test = base.extend<TestFixtures, WorkerFixtures>({
  // Define the worker-level fixture that creates and manages worker-specific auth
  workerAuth: [
    async ({}, use, workerInfo) => {
      const workerId = workerInfo.workerIndex;
      const fileName = path.resolve(
        process.cwd(),
        `.auth/worker-${workerId}.json`,
      );

      // Create directory if it doesn't exist
      const dirPath = path.dirname(fileName);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }

      let auth: TestUser;
      if (fs.existsSync(fileName)) {
        auth = JSON.parse(fs.readFileSync(fileName, "utf-8"));
      } else {
        // Generate new worker-specific test user
        auth = {
          email: `test.worker.${workerId}.${Date.now()}@example.com`,
          password: faker.internet.password({ length: 12 }),
        };

        const supabase = getSupabaseAdmin();
        const {
          data: { user },
          error: signUpError,
        } = await supabase.auth.signUp({
          email: auth.email,
          password: auth.password,
        });

        if (signUpError) {
          throw signUpError;
        }

        auth.id = user?.id;
        fs.writeFileSync(fileName, JSON.stringify(auth));
      }

      await use(auth);

      // Cleanup code is commented out to preserve test users during development
      /*
    if (workerInfo.project.metadata.teardown) {
      if (auth.id) {
        await deleteTestUser(auth.id);
      }
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

  // Update login page fixture to use worker auth by default
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
});

export { expect } from "@playwright/test";
