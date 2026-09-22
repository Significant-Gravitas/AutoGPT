import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";

interface Args {
  enabled: boolean;
}

export function useHomePage({ enabled }: Args) {
  const query = useGetHomeDashboard({
    query: {
      select: (response) => okData(response) ?? null,
      enabled,
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
