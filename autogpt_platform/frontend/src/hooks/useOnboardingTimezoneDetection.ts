import { useEffect, useRef } from "react";
import { usePostV1UpdateUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";

/**
 * Hook to silently detect and set user's timezone during onboarding
 * This version doesn't show any toast notifications
 * @returns void
 */
export const useOnboardingTimezoneDetection = () => {
  const updateTimezone = usePostV1UpdateUserTimezone();
  const hasAttemptedDetection = useRef(false);

  useEffect(() => {
    // Only attempt once
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

        // Silently update the timezone in the backend
        await updateTimezone.mutateAsync({
          data: { timezone: browserTimezone } as any,
        });

        console.log(
          `Timezone automatically set to ${browserTimezone} during onboarding`,
        );
      } catch (error) {
        console.error(
          "Failed to auto-detect timezone during onboarding:",
          error,
        );
        // Silent failure - user can still set timezone manually later
      }
    };

    // Small delay to ensure user is created
    const timer = setTimeout(() => {
      detectAndSetTimezone();
    }, 1000);

    return () => clearTimeout(timer);
  }, []); // Run once on mount
};
