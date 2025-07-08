import { faker } from "@faker-js/faker";
import { chromium, webkit } from "@playwright/test";
import fs from "fs";
import path from "path";
import { signupTestUser } from "./signup";

export interface TestUser {
  email: string;
  password: string;
  id?: string;
  createdAt?: string;
}

export interface UserPool {
  users: TestUser[];
  createdAt: string;
  version: string;
}

// Using Playwright MCP server tools for browser automation
// No need to manage browser instances manually

/**
 * Create a new test user through signup page using Playwright MCP server
 * @param email - User email (optional, will generate if not provided)
 * @param password - User password (optional, will generate if not provided)
 * @param ignoreOnboarding - Skip onboarding and go to marketplace (default: true)
 * @returns Promise<TestUser> - Created user object
 */
export async function createTestUser(
  email?: string,
  password?: string,
  ignoreOnboarding: boolean = true,
): Promise<TestUser> {
  const userEmail = email || faker.internet.email();
  const userPassword = password || faker.internet.password({ length: 12 });

  try {
    const browserType = process.env.BROWSER_TYPE || "chromium";

    const browser =
      browserType === "webkit"
        ? await webkit.launch({ headless: true })
        : await chromium.launch({ headless: true });

    const context = await browser.newContext();
    const page = await context.newPage();

    try {
      const testUser = await signupTestUser(
        page,
        userEmail,
        userPassword,
        ignoreOnboarding,
      );
      return testUser;
    } finally {
      await page.close();
      await context.close();
      await browser.close();
    }
  } catch (error) {
    console.error(`❌ Error creating test user ${userEmail}:`, error);
    throw error;
  }
}

/**
 * Create multiple test users
 * @param count - Number of users to create
 * @returns Promise<TestUser[]> - Array of created users
 */
export async function createTestUsers(count: number): Promise<TestUser[]> {
  console.log(`👥 Creating ${count} test users...`);

  const users: TestUser[] = [];

  for (let i = 0; i < count; i++) {
    try {
      const user = await createTestUser();
      users.push(user);
      console.log(`✅ Created user ${i + 1}/${count}: ${user.email}`);
    } catch (error) {
      console.error(`❌ Failed to create user ${i + 1}/${count}:`, error);
      // Continue creating other users even if one fails
    }
  }

  console.log(`🎉 Successfully created ${users.length}/${count} test users`);
  return users;
}

/**
 * Save user pool to file system
 * @param users - Array of users to save
 * @param filePath - Path to save the file (optional)
 */
export async function saveUserPool(
  users: TestUser[],
  filePath?: string,
): Promise<void> {
  const defaultPath = path.resolve(process.cwd(), ".auth", "user-pool.json");
  const finalPath = filePath || defaultPath;

  // Ensure .auth directory exists
  const dirPath = path.dirname(finalPath);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  const userPool: UserPool = {
    users,
    createdAt: new Date().toISOString(),
    version: "1.0.0",
  };

  try {
    fs.writeFileSync(finalPath, JSON.stringify(userPool, null, 2));
    console.log(`✅ Successfully saved user pool to: ${finalPath}`);
  } catch (error) {
    console.error(`❌ Failed to save user pool to ${finalPath}:`, error);
    throw error;
  }
}

/**
 * Load user pool from file system
 * @param filePath - Path to load from (optional)
 * @returns Promise<UserPool | null> - Loaded user pool or null if not found
 */
export async function loadUserPool(
  filePath?: string,
): Promise<UserPool | null> {
  const defaultPath = path.resolve(process.cwd(), ".auth", "user-pool.json");
  const finalPath = filePath || defaultPath;

  console.log(`📖 Loading user pool from: ${finalPath}`);

  try {
    if (!fs.existsSync(finalPath)) {
      console.log(`⚠️ User pool file not found: ${finalPath}`);
      return null;
    }

    const fileContent = fs.readFileSync(finalPath, "utf-8");
    const userPool: UserPool = JSON.parse(fileContent);

    console.log(
      `✅ Successfully loaded ${userPool.users.length} users from: ${finalPath}`,
    );
    console.log(`📅 User pool created at: ${userPool.createdAt}`);
    console.log(`🔖 User pool version: ${userPool.version}`);

    return userPool;
  } catch (error) {
    console.error(`❌ Failed to load user pool from ${finalPath}:`, error);
    return null;
  }
}

/**
 * Clean up all test users from a pool
 * Note: When using signup page method, cleanup removes the user pool file
 * @param filePath - Path to load from (optional)
 */
export async function cleanupTestUsers(filePath?: string): Promise<void> {
  const defaultPath = path.resolve(process.cwd(), ".auth", "user-pool.json");
  const finalPath = filePath || defaultPath;

  console.log(`🧹 Cleaning up test users...`);

  try {
    if (fs.existsSync(finalPath)) {
      fs.unlinkSync(finalPath);
      console.log(`✅ Deleted user pool file: ${finalPath}`);
    } else {
      console.log(`⚠️ No user pool file found to cleanup`);
    }
  } catch (error) {
    console.error(`❌ Failed to cleanup user pool:`, error);
  }

  console.log(`🎉 Cleanup completed`);
}
