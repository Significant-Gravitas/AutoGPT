import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
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

  useEffect(() => {
    // Only run during actual onboarding routes - prevents running on every auth
    if (!pathname.startsWith("/onboarding")) {
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

        console.log(
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

    // Small delay to ensure user is created and route is stable
    const timer = setTimeout(() => {
      detectAndSetTimezone();
    }, 1000);

    return () => clearTimeout(timer);
  }, [pathname, updateTimezone]); // Include pathname and updateTimezone in deps
};
