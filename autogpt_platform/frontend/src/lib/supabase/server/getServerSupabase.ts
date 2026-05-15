import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { cache } from "react";
import { getCookieSettings } from "../helpers";
import { environment } from "@/services/environment";

type Cookies = { name: string; value: string; options?: CookieOptions }[];

export const getServerSupabase = cache(async () => {
  // Need require here, so Next.js doesn't complain about importing this on client side
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { cookies } = require("next/headers");
  const cookieStore = await cookies();

  return createServerClient(
    environment.getSupabaseUrl(),
    environment.getSupabaseAnonKey(),
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet: Cookies) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, {
                ...options,
                ...getCookieSettings(),
              });
            });
          } catch {
            // The `setAll` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    },
  );
});
