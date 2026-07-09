import { useListBotPlatforms } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";

export function useBotsList() {
  const { data, isLoading, isSuccess, isError, error, refetch } =
    useListBotPlatforms({
      query: { retry: false },
    });

  const platforms = data?.status === 200 ? data.data : [];

  return {
    platforms,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty: isSuccess && platforms.length === 0,
  };
}
