/* eslint-disable react-hooks/rules-of-hooks */
import { createClient, SupabaseClient } from "@supabase/supabase-js";

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
