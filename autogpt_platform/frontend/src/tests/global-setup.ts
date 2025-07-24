import { FullConfig } from "@playwright/test";
import { createTestUsers, saveUserPool, loadUserPool } from "./utils/auth";
import { createAndSaveTestAgents } from "./utils/agent-creation";
/**
 * Global setup function that runs before all tests
 * Creates test users and saves them to file system
 */
async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting global test setup...");
  console.log(`   ⏰ Started at: ${new Date().toLocaleTimeString()}`);
  console.log(`   👷 Workers configured: ${config.workers || 1}`);
  console.log(`   📁 Working directory: ${process.cwd()}`);

  const setupStartTime = Date.now();

  try {
    console.log("\n📋 Phase 1: User Pool Setup");
    console.log("=".repeat(50));

    const existingUserPool = await loadUserPool();

    if (existingUserPool && existingUserPool.users.length > 0) {
      console.log(
        `♻️ Found existing user pool with ${existingUserPool.users.length} users`,
      );
      console.log(`   📅 Pool created at: ${existingUserPool.createdAt}`);
      console.log(`   🔖 Pool version: ${existingUserPool.version}`);
      console.log(
        `   👤 Sample users: ${existingUserPool.users
          .slice(0, 2)
          .map((u) => u.email)
          .join(
            ", ",
          )}${existingUserPool.users.length > 2 ? ` ... and ${existingUserPool.users.length - 2} more` : ""}`,
      );
      console.log("✅ Using existing user pool - skipping user creation");
    } else {
      console.log(
        "📋 No existing user pool found - proceeding with user creation",
      );

    // Create test users using signup page
    const numberOfUsers = (config.workers || 1) + 8; // workers + buffer
    console.log(`👥 Creating ${numberOfUsers} test users via signup...`);
    console.log("⏳ Note: This may take a few minutes in CI environments");

      // Save user pool
      console.log(`🔄 Saving user pool to filesystem...`);
      const saveStartTime = Date.now();
      await saveUserPool(users);
      const saveDuration = Date.now() - saveStartTime;

      console.log(`✅ User creation completed successfully!`);
      console.log(`   📊 Created ${users.length}/${numberOfUsers} test users`);
      console.log(
        `   ⏰ User creation took: ${userCreationDuration}ms (${(userCreationDuration / 1000).toFixed(2)}s)`,
      );
      console.log(`   💾 Save took: ${saveDuration}ms`);
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

    const totalSetupDuration = Date.now() - setupStartTime;
    console.log("\n🎉 Global Setup Summary");
    console.log("=".repeat(50));
    console.log(`✅ Global setup completed successfully!`);
    console.log(
      `   ⏰ Total setup duration: ${totalSetupDuration}ms (${(totalSetupDuration / 1000).toFixed(2)}s)`,
    );
    console.log(`   🏁 Completed at: ${new Date().toLocaleTimeString()}`);
    console.log(`   📈 Ready for ${config.workers || 1} parallel workers`);
    console.log("=".repeat(50));
  } catch (error) {
    const totalSetupDuration = Date.now() - setupStartTime;
    console.error("\n❌ Global Setup Failed");
    console.error("=".repeat(50));
    console.error("❌ Global setup failed:", error);
    console.error("💡 This is likely due to:");
    console.error("   1. Backend services not fully ready");
    console.error("   2. Network timeouts in CI environment");
    console.error("   3. Database or authentication issues");
    throw error;
  }
}

export default globalSetup;
