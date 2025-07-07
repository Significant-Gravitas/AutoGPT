import { FullConfig } from "@playwright/test";
import { createTestUsers, saveUserPool, loadUserPool } from "./utils/auth";

/**
 * Global setup function that runs before all tests
 * Creates test users and saves them to file system
 */
async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting global test setup...");

  try {
    const existingUserPool = await loadUserPool();

    if (existingUserPool && existingUserPool.users.length > 0) {
      console.log(
        `♻️ Found existing user pool with ${existingUserPool.users.length} users`,
      );
      console.log("✅ Using existing user pool");
      return;
    }

    // Create test users using signup page
    const numberOfUsers = (config.workers || 1) + 3; // workers + buffer
    console.log(`👥 Creating ${numberOfUsers} test users via signup...`);

    const users = await createTestUsers(numberOfUsers);

    if (users.length === 0) {
      throw new Error("Failed to create any test users");
    }

    // Save user pool
    await saveUserPool(users);

    console.log("✅ Global setup completed successfully!");
    console.log(`📊 Created ${users.length} test users via signup page`);
  } catch (error) {
    console.error("❌ Global setup failed:", error);
    throw error;
  }
}

export default globalSetup;
