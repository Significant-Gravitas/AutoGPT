"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createBrowserClient } from "@supabase/ssr";
import { User } from "@supabase/supabase-js";

import { _logoutServer } from "./actions";

export default function useSupabase() {
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
    () => Promise.all([_logoutServer(), supabase?.auth.signOut()]),
    [supabase],
  );

  return { supabase, user, isLoggedIn: !!user, isUserLoading, logOut };
}
