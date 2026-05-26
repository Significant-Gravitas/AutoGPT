import { useListBotPlatforms } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";

export function useBotsList() {
  const { data, isLoading, isError, error, refetch } = useListBotPlatforms({
    query: { retry: false },
  });

  const platforms = data?.status === 200 ? data.data : [];

  return {
    platforms,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty: !isLoading && !isError && platforms.length === 0,
  };
}
