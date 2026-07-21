import { useListBotPlatforms } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function useBotsList() {
  const { data, isLoading, isSuccess, isError, error, refetch } =
    useListBotPlatforms({
      query: { retry: false },
    });
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
