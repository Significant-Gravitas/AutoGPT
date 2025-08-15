import { FullConfig } from "@playwright/test";
import { createTestUsers, saveUserPool, loadUserPool } from "./utils/auth";

async function globalSetup(config: FullConfig) {
  try {
    const existingUserPool = await loadUserPool();

    if (existingUserPool && existingUserPool.users.length > 0) {
      return;
    }

    // Create test users using signup page
    const numberOfUsers = (config.workers || 1) + 8; // workers + buffer

    const users = await createTestUsers(numberOfUsers);

    if (users.length === 0) {
      throw new Error("Failed to create any test users");
    }

    // Require at least a minimum number of users for tests to work
    const minUsers = Math.max(config.workers || 1, 2);
    if (users.length < minUsers) {
      throw new Error(
        `Only created ${users.length} users but need at least ${minUsers} for tests to run properly`,
      );
    }

    // Save user pool
    await saveUserPool(users);
  } catch (error) {
    throw error;
  }
}

export default globalSetup;
