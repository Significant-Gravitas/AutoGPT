import { createServerClient } from "@supabase/ssr";

export default function getServerSupabase() {
  // Need require here, so Next.js doesn't complain about importing this on client side
  const { cookies } = require("next/headers");
  const cookieStore = cookies();

  try {
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll(cookiesToSet) {
            try {
              cookiesToSet.forEach(({ name, value, options }) =>
                cookieStore.set(name, value, options),
              );
            } catch {
              // The `setAll` method was called from a Server Component.
              // This can be ignored if you have middleware refreshing
              // user sessions.
            }
          },
        },
      },
    );

    return supabase;
  } catch (error) {
    throw error;
  }
}
