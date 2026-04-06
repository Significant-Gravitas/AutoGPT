import { FullConfig } from "@playwright/test";
import { ensureSeededAuthStates, hasSeededAuthStates } from "./utils/auth";

function resolveBaseURL(config: FullConfig) {
  const configuredBaseURL =
    config.projects[0]?.use?.baseURL ?? "http://localhost:3000";

  if (typeof configuredBaseURL !== "string") {
    throw new Error(
      `Playwright baseURL must be a string during global setup. Received ${String(
        configuredBaseURL,
      )}.`,
    );
  }

  return configuredBaseURL;
}

async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting global test setup...");

  try {
    const baseURL = resolveBaseURL(config);

    if (hasSeededAuthStates()) {
      console.log("♻️ Reusing stored seeded auth states");
      return;
    }

    console.log("🔐 Creating reusable seeded auth states...");
    await ensureSeededAuthStates(baseURL);

    console.log("✅ Global setup completed successfully!");
  } catch (error) {
    console.error("❌ Global setup failed:", error);
    console.error(
      "💡 Run backend/test/e2e_test_data.py to seed the deterministic Playwright accounts before retrying.",
    );
    throw error;
  }
}

export default globalSetup;
