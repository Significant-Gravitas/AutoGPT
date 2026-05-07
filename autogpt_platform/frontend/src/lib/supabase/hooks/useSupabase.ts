"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useShallow } from "zustand/react/shallow";
import type { ServerLogoutOptions } from "../actions";
import { useSupabaseStore } from "./useSupabaseStore";

export function useSupabase() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const api = useBackendAPI();

  const search = searchParams.toString();
  const fullPath = search ? `${pathname}?${search}` : pathname;

  const {
    user,
    supabase,
    isUserLoading,
    initialize,
    setCurrentRequestContext,
    logOut,
    validateSession,
    refreshSession,
  } = useSupabaseStore(
    useShallow((state) => ({
      user: state.user,
      supabase: state.supabase,
      isUserLoading: state.isUserLoading,
      initialize: state.initialize,
      setCurrentRequestContext: state.setCurrentRequestContext,
      logOut: state.logOut,
      validateSession: state.validateSession,
      refreshSession: state.refreshSession,
    })),
  );

  useMountEffect(() => {
    void initialize({ api, router, path: fullPath });
  });

  // Keep the store's request context in sync with navigation. Safe to call
  // during render because `useSupabase` doesn't subscribe to the fields that
  // `setCurrentRequestContext` writes (routerRef / apiRef / currentPath).
  setCurrentRequestContext({ api, router, path: fullPath });

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
