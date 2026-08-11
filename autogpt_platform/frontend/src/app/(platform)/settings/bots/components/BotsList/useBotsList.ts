import { useListBotPlatforms } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect } from "react";

export function useBotsList() {
  const { data, isLoading, isSuccess, isError, error, refetch } =
    useListBotPlatforms({
      query: {
        retry: false,
        // This page mirrors state changed outside the app — a workspace
        // install, or linking from inside the chat client. The global 60s
        // staleTime would serve cache to someone returning from exactly that
        // round-trip, so always refetch on mount/focus instead.
        staleTime: 0,
        refetchOnWindowFocus: true,
        refetchOnMount: "always",
      },
    });

  // The install flow ends in Slack, not back on this page, so the only signal
  // that the user has returned is the tab regaining visibility. Subscribe
  // directly rather than relying on the query client's focus handling, which
  // doesn't fire reliably when the trip out went through a native app.
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
