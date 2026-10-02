import { useGetV2ListChatTransports } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useUpdateExpertLlmRoute,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import {
  getRouteNote,
  parseRouteValue,
  routeOptions,
  routeValue,
} from "./helpers";

interface Args {
  expert: Expert;
}

export function useExpertLlmRouteSection({ expert }: Args) {
  const queryClient = useQueryClient();

  const transportsQuery = useGetV2ListChatTransports({
    query: {
      refetchOnWindowFocus: true,
      select: (response) =>
        response.status === 200 ? response.data.transports : [],
    },
  });
  const credentialsQuery = useGetV1ListCredentials({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  const { mutate, isPending } = useUpdateExpertLlmRoute({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expert.id),
        });
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
        toast({ title: "AI connection updated" });
      },
      onError: () => {
        toast({
          title: "Could not update the AI connection",
          description:
            "Check the connection is still linked in Settings, then try again.",
          variant: "destructive",
        });
      },
    },
  });

  function selectRoute(value: string) {
    if (value === routeValue(expert) || isPending) return;
    mutate({ expertId: expert.id, data: parseRouteValue(value) });
  }

  return {
    value: routeValue(expert),
    options: routeOptions(
      transportsQuery.data ?? [],
      credentialsQuery.data ?? [],
      expert,
    ),
    note: getRouteNote(expert),
    isLoading: transportsQuery.isLoading,
    isSaving: isPending,
    selectRoute,
  };
}
