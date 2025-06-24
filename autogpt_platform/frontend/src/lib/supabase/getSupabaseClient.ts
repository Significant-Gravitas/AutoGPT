import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { createBrowserClient } from "@supabase/ssr";

const isClient = typeof window !== "undefined";

export const getSupabaseClient = async () => {
  return isClient
    ? createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        { isSingleton: true },
      )
    : await getServerSupabase();
};
