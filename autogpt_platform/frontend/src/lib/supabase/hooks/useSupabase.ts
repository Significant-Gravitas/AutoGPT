"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import type { ServerLogoutOptions } from "../actions";
import { useSupabaseStore } from "./useSupabaseStore";

export function useSupabase() {
  const router = useRouter();
  const pathname = usePathname();
  const api = useBackendAPI();

  const {
    user,
    supabase,
    isUserLoading,
    initialize,
    logOut,
    validateSession,
    refreshSession,
  } = useSupabaseStore(
    useShallow((state) => ({
      user: state.user,
      supabase: state.supabase,
      isUserLoading: state.isUserLoading,
      initialize: state.initialize,
      logOut: state.logOut,
      validateSession: state.validateSession,
      refreshSession: state.refreshSession,
    })),
  );

  useEffect(() => {
    void initialize({
      api,
      router,
      pathname,
    });
  }, [api, initialize, pathname, router]);

  function handleLogout(options: ServerLogoutOptions = {}) {
    return logOut({
      options,
      api,
    });
  }

  function handleValidateSession() {
    return validateSession({
      pathname,
      router,
    });
  }

  return {
    user,
    supabase,
    isLoggedIn: Boolean(user),
    isUserLoading,
    logOut: handleLogout,
    validateSession: handleValidateSession,
    refreshSession,
  };
}
