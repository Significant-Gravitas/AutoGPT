/* eslint-disable react-hooks/rules-of-hooks */
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { faker } from "@faker-js/faker";

export type TestUser = {
  email: string;
  password: string;
  id?: string;
};

let supabase: SupabaseClient;

function getSupabaseAdmin() {
  if (!supabase) {
    supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false,
        },
      },
    );
  }
  return supabase;
}

async function createTestUser(userData: TestUser): Promise<TestUser> {
  const supabase = getSupabaseAdmin();

  const { data: authUser, error: authError } = await supabase.auth.signUp({
    email: userData.email,
    password: userData.password,
  });

  if (authError) {
    throw new Error(`Failed to create test user: ${authError.message}`);
  }

  return {
    ...userData,
    id: authUser.user?.id,
  };
}

async function deleteTestUser(userId: string) {
  const supabase = getSupabaseAdmin();

  try {
    const { error } = await supabase.auth.admin.deleteUser(userId);

    if (error) {
      console.warn(`Warning: Failed to delete test user: ${error.message}`);
    }
  } catch (error) {
    console.warn(
      `Warning: Error during user cleanup: ${(error as Error).message}`,
    );
  }
}

function generateUserData(): TestUser {
  return {
    email: `test.${faker.string.uuid()}@example.com`,
    password: faker.internet.password({ length: 12 }),
  };
}

// Export just the fixture function
export const createTestUserFixture = async ({}, use) => {
  let user: TestUser | null = null;

  try {
    const userData = generateUserData();
    user = await createTestUser(userData);
    await use(user);
  } finally {
    if (user?.id) {
      await deleteTestUser(user.id);
    }
  }
};
