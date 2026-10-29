import { useListBotPlatforms } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect } from "react";

export function useBotsList() {
  const { data, isLoading, isSuccess, isError, error, refetch } =
    useListBotPlatforms({
      query: {
        retry: false,
        // Links are made outside the app, so the global 60s staleTime would
        // serve cache to someone returning from exactly that round-trip.
        staleTime: 0,
        // The listeners below cover coming back; this would double it up.
        refetchOnWindowFocus: false,
        refetchOnMount: "always",
      },
    });

  // Both events are needed: tab switches fire visibilitychange, while
  // alt-tabbing back from the desktop app fires focus on a still-visible tab.
  useEffect(() => {
    function refetchIfVisible() {
      if (document.visibilityState === "visible") refetch();
    }

    window.addEventListener("focus", refetchIfVisible);
    document.addEventListener("visibilitychange", refetchIfVisible);
    return () => {
      window.removeEventListener("focus", refetchIfVisible);
      document.removeEventListener("visibilitychange", refetchIfVisible);
    };
  }, [refetch]);
  const visibility = useGetFlag(Flag.COPILOT_BOT_PLATFORMS);

  const allPlatforms = data?.status === 200 ? data.data : [];
  // Only an explicit false hides a platform, so a missing flag key (or a
  // LaunchDarkly outage) fails open to visible.
  const platforms = allPlatforms.filter(
    (platform) => visibility[platform.platform.toLowerCase()] !== false,
  );

  return {
    platforms,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty: isSuccess && platforms.length === 0,
  };
}
