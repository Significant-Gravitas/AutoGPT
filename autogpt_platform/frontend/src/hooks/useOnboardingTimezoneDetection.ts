import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { usePostV1UpdateUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";

/**
 * Hook to silently detect and set user's timezone ONLY during actual onboarding flow
 * This prevents unnecessary timezone API calls during authentication and platform usage
 * @returns void
 */
export const useOnboardingTimezoneDetection = () => {
  const updateTimezone = usePostV1UpdateUserTimezone();
  const hasAttemptedDetection = useRef(false);
  const pathname = usePathname();
  const { user, isUserLoading } = useSupabase();

  // Check if we're on onboarding route (computed outside useEffect to avoid re-computing)
  const isOnOnboardingRoute = pathname.startsWith("/onboarding");

  useEffect(() => {
    // Only run during actual onboarding routes - prevents running on every auth
    if (!isOnOnboardingRoute) {
      return;
    }

    // Wait for proper authentication state instead of using arbitrary timeout
    if (isUserLoading || !user) {
      return;
    }

    // Only attempt once per session
    if (hasAttemptedDetection.current) {
      return;
    }

    const detectAndSetTimezone = async () => {
      // Mark that we've attempted detection
      hasAttemptedDetection.current = true;

      try {
        // Detect browser timezone
        const browserTimezone =
          Intl.DateTimeFormat().resolvedOptions().timeZone;

        if (!browserTimezone) {
          console.error("Could not detect browser timezone during onboarding");
          return;
        }

        // Fire-and-forget timezone update - we don't need to wait for response
        updateTimezone.mutate({
          data: { timezone: browserTimezone } as any,
        });

        console.info(
          `Timezone automatically set to ${browserTimezone} during onboarding flow`,
        );
      } catch (error) {
        console.error(
          "Failed to auto-detect timezone during onboarding:",
          error,
        );
        // Silent failure - user can still set timezone manually later
      }
    };

    detectAndSetTimezone();
  }, [isOnOnboardingRoute, updateTimezone, user, isUserLoading]); // Use computed boolean to reduce re-renders
};
