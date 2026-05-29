"use client";

import { useEffect } from "react";
import * as Sentry from "@sentry/nextjs";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

/**
 * SentryUserTracker component sets user context in Sentry for error tracking.
 * This component should be placed high in the component tree to ensure user
 * context is available for all error reports.
 *
 * It automatically:
 * - Sets user context when a user logs in
 * - Clears user context when a user logs out
 * - Updates context when user data changes
 */
export function SentryUserTracker() {
  const { user, isUserLoading } = useSupabase();

  useEffect(() => {
    if (user) {
      // Wait until user loading is complete before setting user context
      if (isUserLoading) return;

      // Set user context for Sentry error tracking
      Sentry.setUser({
        id: user.id,
        email: user.email ?? undefined,
        // Add custom attributes
        ...(user.role && { role: user.role }),
      });
    } else {
      // Always clear user context when user is null, regardless of loading state
      // This ensures logout properly clears the context immediately
      Sentry.setUser(null);
    }
  }, [user, isUserLoading]);

  // This component doesn't render anything
  return null;
}
