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
  const [role, setRole] = useState<string | null>(null);

  useEffect(() => {
    if (isSupabaseLoading || !supabase) {
      return;
    }

    const fetchUser = async () => {
      try {
        setIsLoading(true);
        const { data: userData, error: userError } =
          await supabase.auth.getUser();
        const { data: sessionData, error: sessionError } =
          await supabase.auth.getSession();

        if (userError) throw new Error(`User error: ${userError.message}`);
        if (sessionError)
          throw new Error(`Session error: ${sessionError.message}`);

        setUser(userData.user);
        setSession(sessionData.session);
        setRole(userData.user?.role || null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch user data");
        console.error("Error in useUser hook:", e);
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
      setRole(session?.user?.role || null);

      setIsLoading(false);
    });

    return () => subscription.unsubscribe();
  }, [supabase, isSupabaseLoading]);

  return {
    user,
    session,
    role,
    isLoading: isLoading || isSupabaseLoading,
    error,
  };
};

export default useUser;
