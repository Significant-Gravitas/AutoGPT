"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import type { ServerLogoutOptions } from "../actions";
import { useSupabaseStore } from "./useSupabaseStore";

export function useSupabase() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const api = useBackendAPI();

  // Combine pathname and search params to get full path for redirect preservation
  const fullPath = useMemo(() => {
    const search = searchParams.toString();
    return search ? `${pathname}?${search}` : pathname;
  }, [pathname, searchParams]);

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
      path: fullPath,
    });
  }, [api, initialize, fullPath, router]);

  function handleLogout(options: ServerLogoutOptions = {}) {
    return logOut({
      options,
      api,
    });
  }

  function handleValidateSession() {
    return validateSession({
      path: fullPath,
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
