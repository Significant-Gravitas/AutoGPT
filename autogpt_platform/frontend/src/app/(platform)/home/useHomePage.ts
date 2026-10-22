import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";

export function useHomePage({ enabled = true }: { enabled?: boolean } = {}) {
  const query = useGetHomeDashboard({
    query: {
      enabled,
      select: (response) => okData(response) ?? null,
      staleTime: 0,
      refetchInterval: 60_000,
      refetchOnWindowFocus: true,
    },
  });

  return {
    dashboard: query.data ?? null,
    isLoading: enabled && query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
}
