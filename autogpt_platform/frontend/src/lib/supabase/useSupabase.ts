"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createBrowserClient } from "@supabase/ssr";
import { SignOut, User } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";

export default function useSupabase() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [isUserLoading, setIsUserLoading] = useState(true);

  const supabase = useMemo(() => {
    try {
      return createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        { isSingleton: true },
      );
    } catch (error) {
      console.error("Error creating Supabase client", error);
      return null;
    }
  }, []);

  useEffect(() => {
    if (!supabase) {
      setIsUserLoading(false);
      return;
    }

    // Sync up the current state and listen for changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      setIsUserLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase]);

  const logOut = useCallback(
    async (options?: SignOut) => {
      if (!supabase) return;

      const { error } = await supabase.auth.signOut({
        scope: options?.scope ?? "local",
      });
      if (error) console.error("Error logging out:", error);

      router.push("/login");
    },
    [router, supabase],
  );

  if (!supabase || isUserLoading) {
    return { supabase, user: null, isLoggedIn: null, isUserLoading, logOut };
  }
  if (!user) {
    return { supabase, user, isLoggedIn: false, isUserLoading, logOut };
  }
  return { supabase, user, isLoggedIn: true, isUserLoading, logOut };
}
