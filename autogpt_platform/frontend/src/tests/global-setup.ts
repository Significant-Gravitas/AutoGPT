import { FullConfig } from "@playwright/test";
import { createTestUsers, saveUserPool, loadUserPool } from "./utils/auth";

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
    const numberOfUsers = (config.workers || 1) + 8; // workers + buffer
    console.log(`👥 Creating ${numberOfUsers} test users via signup...`);
    console.log("⏳ Note: This may take a few minutes in CI environments");

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

    console.log("✅ Global setup completed successfully!");
    console.log(`📊 Created ${users.length} test users via signup page`);
  } catch (error) {
    console.error("❌ Global setup failed:", error);
    console.error("💡 This is likely due to:");
    console.error("   1. Backend services not fully ready");
    console.error("   2. Network timeouts in CI environment");
    console.error("   3. Database or authentication issues");
    throw error;
  }
}

export default globalSetup;
