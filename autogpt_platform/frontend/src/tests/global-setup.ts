import { chromium, FullConfig } from "@playwright/test";
import fs from "fs";
import path from "path";
import { createAndSetupTestUser } from "./utils/auth";

async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting global test setup...");

  // Create auth directory if it doesn't exist
  const authDir = path.resolve(process.cwd(), ".auth");
  if (!fs.existsSync(authDir)) {
    fs.mkdirSync(authDir, { recursive: true });
  }

  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Create test users for each worker
    const workerCount = config.workers || 1;
    console.log(
      `📊 Creating ${workerCount} test users for parallel workers...`,
    );

    for (let workerId = 0; workerId < workerCount; workerId++) {
      const fileName = path.resolve(authDir, `worker-${workerId}.json`);

      // Skip if user already exists
      if (fs.existsSync(fileName)) {
        console.log(`⏭️  Worker ${workerId} user already exists, skipping...`);
        continue;
      }

      console.log(`👤 Creating user for worker ${workerId}...`);
      const testUser = await createAndSetupTestUser(page);

      // Save user credentials
      fs.writeFileSync(fileName, JSON.stringify(testUser, null, 2));
      console.log(`💾 Saved user for worker ${workerId}: ${testUser.email}`);
    }

    console.log("✅ Global test setup completed successfully!");
  } catch (error) {
    console.error("❌ Global test setup failed:", error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;
