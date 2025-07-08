import { faker } from "@faker-js/faker";
import { TestUser } from "./auth";

/**
 * Generate a test user with random data
 * @param options - Optional parameters to override defaults
 * @returns TestUser object with generated data
 */
export function generateUser(options?: {
  email?: string;
  password?: string;
  name?: string;
}): TestUser {
  console.log("🎲 Generating test user...");

  const user: TestUser = {
    email: options?.email || faker.internet.email(),
    password: options?.password || faker.internet.password({ length: 12 }),
    createdAt: new Date().toISOString(),
  };

  console.log(`✅ Generated user: ${user.email}`);
  return user;
}

/**
 * Generate multiple test users
 * @param count - Number of users to generate
 * @returns Array of TestUser objects
 */
export function generateUsers(count: number): TestUser[] {
  console.log(`👥 Generating ${count} test users...`);

  const users: TestUser[] = [];

  for (let i = 0; i < count; i++) {
    users.push(generateUser());
  }

  console.log(`✅ Generated ${users.length} test users`);
  return users;
}
