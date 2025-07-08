"use client";
import { createBrowserClient } from "@supabase/ssr";
import { User } from "@supabase/supabase-js";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  getCurrentUser,
  refreshSession,
  serverLogout,
  ServerLogoutOptions,
  validateSession,
} from "../actions";
import {
  broadcastLogout,
  getRedirectPath,
  isLogoutEvent,
  setupSessionEventListeners,
} from "../helpers";

export function useSupabase() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [isUserLoading, setIsUserLoading] = useState(true);
  const lastValidationRef = useRef<number>(0);
  const isValidatingRef = useRef(false);

  const supabase = useMemo(() => {
    try {
      return createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        {
          isSingleton: true,
          auth: {
            persistSession: false, // Don't persist session on client with httpOnly cookies
          },
        },
      );
    } catch (error) {
      console.error("Error creating Supabase client", error);
      return null;
    }
  }, []);

  async function logOut(options: ServerLogoutOptions = {}) {
    broadcastLogout();

    try {
      await serverLogout(options);
    } catch (error) {
      console.error("Error logging out:", error);
      router.push("/login");
    }
  }

  async function validateSessionServer() {
    // Prevent concurrent validation calls
    if (isValidatingRef.current) return true;

    // Simple debounce - only validate if 2 seconds have passed
    const now = Date.now();
    if (now - lastValidationRef.current < 2000) {
      return true;
    }

    isValidatingRef.current = true;
    lastValidationRef.current = now;

    try {
      const result = await validateSession(pathname);

      if (!result.isValid) {
        setUser(null);
        if (result.redirectPath) {
          router.push(result.redirectPath);
        }
        return false;
      }

      // Update local state with server user data
      if (result.user) {
        setUser((currentUser) => {
          // Only update if user actually changed to prevent unnecessary re-renders
          if (currentUser?.id !== result.user?.id) {
            return result.user;
          }
          return currentUser;
        });
      }

      return true;
    } catch (error) {
      console.error("Session validation error:", error);
      setUser(null);
      const redirectPath = getRedirectPath(pathname);
      if (redirectPath) {
        router.push(redirectPath);
      }
      return false;
    } finally {
      isValidatingRef.current = false;
    }
  }

  async function getUserFromServer() {
    try {
      const { user: serverUser, error } = await getCurrentUser();

      if (error || !serverUser) {
        setUser(null);
        return null;
      }

      setUser(serverUser);
      return serverUser;
    } catch (error) {
      console.error("Get user error:", error);
      setUser(null);
      return null;
    }
  }

  function handleCrossTabLogout(e: StorageEvent) {
    if (!isLogoutEvent(e)) return;

    // Clear local state immediately
    setUser(null);
    router.refresh();

    const redirectPath = getRedirectPath(pathname);
    if (redirectPath) {
      router.push(redirectPath);
    }
  }

  function handleVisibilityChange() {
    if (document.visibilityState === "visible") {
      validateSessionServer();
    }
  }

  function handleFocus() {
    validateSessionServer();
  }

  useEffect(() => {
    getUserFromServer().finally(() => {
      setIsUserLoading(false);
    });

    // Set up event listeners for cross-tab logout, focus, and visibility change
    const eventListeners = setupSessionEventListeners(
      handleVisibilityChange,
      handleFocus,
      handleCrossTabLogout,
    );

    return () => {
      eventListeners.cleanup();
    };
  }, []);

  return {
    supabase, // Available for non-auth operations like real-time subscriptions
    user,
    isLoggedIn: !isUserLoading ? !!user : null,
    isUserLoading,
    logOut,
    validateSession: validateSessionServer,
    refreshSession,
  };
}
