"use client";
import { useEffect, useMemo, useState, useRef } from "react";
import { createBrowserClient } from "@supabase/ssr";
import { User } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import {
  broadcastLogout,
  getRedirectPath,
  isLogoutEvent,
  setupSessionEventListeners,
} from "../helpers";

export function useSupabase() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [isUserLoading, setIsUserLoading] = useState(true);
  const lastValidationRef = useRef<number>(0);

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

  async function logOut() {
    if (!supabase) return;

    broadcastLogout();

    const { error } = await supabase.auth.signOut({
      scope: "global",
    });
    if (error) console.error("Error logging out:", error);

    router.push("/login");
  }

  async function validateSession() {
    if (!supabase) return false;

    // Simple debounce - only validate if 2 seconds have passed
    const now = Date.now();
    if (now - lastValidationRef.current < 2000) {
      return true;
    }
    lastValidationRef.current = now;

    try {
      const {
        data: { user: apiUser },
        error,
      } = await supabase.auth.getUser();

      if (error || !apiUser) {
        // Session is invalid, clear local state
        setUser(null);
        const redirectPath = getRedirectPath(window.location.pathname);
        if (redirectPath) {
          router.push(redirectPath);
        }
        return false;
      }

      // Update local state if we have a valid user but no local user
      if (apiUser && !user) {
        setUser(apiUser);
      }

      return true;
    } catch (error) {
      console.error("Session validation error:", error);
      setUser(null);
      const redirectPath = getRedirectPath(window.location.pathname);
      if (redirectPath) {
        router.push(redirectPath);
      }
      return false;
    }
  }

  function handleCrossTabLogout(e: StorageEvent) {
    if (!isLogoutEvent(e)) return;

    // Clear the Supabase session first
    if (supabase) {
      supabase.auth.signOut({ scope: "global" }).catch(console.error);
    }

    // Clear local state immediately
    setUser(null);
    router.refresh();

    const redirectPath = getRedirectPath(window.location.pathname);
    if (redirectPath) {
      router.push(redirectPath);
    }
  }

  function handleVisibilityChange() {
    if (document.visibilityState === "visible") {
      validateSession();
    }
  }

  function handleFocus() {
    validateSession();
  }

  useEffect(() => {
    if (!supabase) {
      setIsUserLoading(false);
      return;
    }

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_, session) => {
      const newUser = session?.user ?? null;

      // Only update if user actually changed to prevent unnecessary re-renders
      setUser((currentUser) => {
        if (currentUser?.id !== newUser?.id) {
          return newUser;
        }
        return currentUser;
      });

      setIsUserLoading(false);
    });

    const eventListeners = setupSessionEventListeners(
      handleVisibilityChange,
      handleFocus,
      handleCrossTabLogout,
    );

    return () => {
      subscription.unsubscribe();
      eventListeners.cleanup();
    };
  }, [supabase]);

  return {
    supabase,
    user,
    isLoggedIn: !isUserLoading ? !!user : null,
    isUserLoading,
    logOut,
    validateSession,
  };
}
