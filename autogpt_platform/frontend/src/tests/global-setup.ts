import { chromium, firefox, FullConfig, webkit } from "@playwright/test";
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

  const baseURL = "http://localhost:3000/";
  console.log(`🌐 Using base URL: ${baseURL}`);

  const browserName = config.projects[0]?.use?.browserName || "chromium";
  console.log(`🌐 Using browser: ${browserName}`);

  const browserType = {
    chromium: chromium,
    webkit: webkit,
    firefox: firefox,
  }[browserName];

  if (!browserType) {
    throw new Error(`Unsupported browser: ${browserName}`);
  }

  const browser = await browserType.launch();

  try {
    // Wait for the server to be ready with retry logic
    console.log("🔄 Waiting for server to be ready...");
    const context = await browser.newContext({
      baseURL: baseURL,
    });
    const page = await context.newPage();

    let serverReady = false;
    let retries = 0;
    const maxRetries = 60; // 60 seconds timeout

    while (!serverReady && retries < maxRetries) {
      try {
        await page.goto("/", { waitUntil: "domcontentloaded", timeout: 10000 });
        serverReady = true;
        console.log("✅ Server is ready");
      } catch (error) {
        retries++;
        console.log(
          `🔄 Server not ready, retrying... (${retries}/${maxRetries})`,
        );
        console.log(`Error: ${error}`);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    if (!serverReady) {
      throw new Error("Server failed to start within timeout period");
    }

    await context.close();

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
      try {
        // Create a fresh context for each user
        const userContext = await browser.newContext({
          baseURL: baseURL,
        });
        const userPage = await userContext.newPage();

        const testUser = await createAndSetupTestUser(userPage);

        // Save user credentials
        fs.writeFileSync(fileName, JSON.stringify(testUser, null, 2));
        console.log(
          `💾 Saved user for worker ${workerId}: ${testUser.email} at ${fileName}`,
        );

        await userContext.close();
      } catch (userError) {
        console.error(
          `❌ Failed to create user for worker ${workerId}:`,
          userError,
        );
        throw userError;
      }
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
