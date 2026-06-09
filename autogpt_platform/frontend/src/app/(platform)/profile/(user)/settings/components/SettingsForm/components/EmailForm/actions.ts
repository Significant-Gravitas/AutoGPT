import { getServerSupabase } from "@/lib/auth/server/getServerSupabase";

export async function updateSupabaseUserEmail(email: string) {
  const supabase = await getServerSupabase();
  const { data, error } = await supabase.auth.updateUser({
    email,
  });

  return { data, error };
}
