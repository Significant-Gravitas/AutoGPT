"use client";

import { useEffect, useState } from "react";
import { User, Session } from "@supabase/supabase-js";
import { useSupabase } from "@/components/SupabaseProvider";

const useUser = () => {
  const { supabase, isLoading: isSupabaseLoading } = useSupabase();
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isSupabaseLoading || !supabase) {
      return;
    }

    const fetchUser = async () => {
      try {
        setIsLoading(true);
        const {
          data: { user },
        } = await supabase.auth.getUser();
        const {
          data: { session },
        } = await supabase.auth.getSession();
        setUser(user);
        setSession(session);
      } catch (e) {
        setError("Failed to fetch user data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchUser();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      setIsLoading(false);
    });

    return () => subscription.unsubscribe();
  }, [supabase, isSupabaseLoading]);

  return { user, session, isLoading: isLoading || isSupabaseLoading, error };
};

export default useUser;
