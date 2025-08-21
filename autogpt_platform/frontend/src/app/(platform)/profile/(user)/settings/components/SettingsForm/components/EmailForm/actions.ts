import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";

export async function updateSupabaseUserEmail(email: string) {
  const supabase = await getServerSupabase();
  const { data, error } = await supabase.auth.updateUser({
    email,
  });

  return { data, error };
}
