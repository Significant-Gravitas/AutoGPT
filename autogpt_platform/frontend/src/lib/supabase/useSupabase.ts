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

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    supabase.auth.getUser().then((response) => {
      if (response.error) {
        // Display error only if it's not about missing auth session (user is not logged in)
        if (response.error.message !== "Auth session missing!") {
          console.error("Error fetching user", response.error);
        }
        setUser(null);
      } else {
        setUser(response.data.user);
      }
      setIsUserLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase]);

  const logOut = useCallback(() => {
    supabase?.auth.signOut().then(() => _logoutServer());
  }, [supabase]);

  return { supabase, user, isLoggedIn: !!user, isUserLoading, logOut };
}
