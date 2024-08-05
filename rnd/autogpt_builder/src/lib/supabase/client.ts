import { createBrowserClient } from "@supabase/ssr";

export function createClient() {
  try {
    return createBrowserClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    );
  } catch (error) {
    return null;
  }
}
