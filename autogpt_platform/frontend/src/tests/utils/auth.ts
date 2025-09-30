import fs from "fs";
import path from "path";
import { signupTestUser } from "./signup";
import { getBrowser } from "./get-browser";

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

export async function createTestUser(
  email?: string,
  password?: string,
  ignoreOnboarding: boolean = true,
): Promise<TestUser> {
  const { faker } = await import("@faker-js/faker");
  const userEmail = email || faker.internet.email();
  const userPassword = password || faker.internet.password({ length: 12 });

  try {
    const browser = await getBrowser();
    const context = await browser.newContext();
    const page = await context.newPage();

    try {
      const testUser = await signupTestUser(
        page,
        userEmail,
        userPassword,
        ignoreOnboarding,
        false,
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

export async function createTestUsers(count: number): Promise<TestUser[]> {
  console.log(`👥 Creating ${count} test users...`);

  const users: TestUser[] = [];
  let consecutiveFailures = 0;

  for (let i = 0; i < count; i++) {
    try {
      const user = await createTestUser();
      users.push(user);
      consecutiveFailures = 0; // Reset failure counter on success
      console.log(`✅ Created user ${i + 1}/${count}: ${user.email}`);

      // Small delay to prevent overwhelming the system
      if (i < count - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      consecutiveFailures++;
      console.error(`❌ Failed to create user ${i + 1}/${count}:`, error);

      // If we have too many consecutive failures, stop trying
      if (consecutiveFailures >= 3) {
        console.error(
          `⚠️ Stopping after ${consecutiveFailures} consecutive failures`,
        );
        break;
      }

      // Add a longer delay after failure to let system recover
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  console.log(`🎉 Successfully created ${users.length}/${count} test users`);
  return users;
}

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

export async function getTestUser(): Promise<TestUser> {
  const userPool = await loadUserPool();
  if (!userPool) {
    throw new Error("User pool not found");
  }

  if (userPool.users.length === 0) {
    throw new Error("No users available in the pool");
  }

  // Return a random user from the pool
  const randomIndex = Math.floor(Math.random() * userPool.users.length);
  return userPool.users[randomIndex];
}
