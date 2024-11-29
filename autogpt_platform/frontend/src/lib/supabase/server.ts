import {
  createServerClient as createClient,
  type CookieOptions,
} from "@supabase/ssr";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

export function createServerClient() {
  const cookieStore = cookies();

  try {
    return createClient(
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
  } catch (error) {
    return null;
  }
}

export async function checkAuth() {
  const supabase = createServerClient();
  if (!supabase) {
    console.error("No supabase client");
    redirect("/login");
  }
  const { data, error } = await supabase.auth.getUser();
  if (error || !data?.user) {
    redirect("/login");
  }
}
