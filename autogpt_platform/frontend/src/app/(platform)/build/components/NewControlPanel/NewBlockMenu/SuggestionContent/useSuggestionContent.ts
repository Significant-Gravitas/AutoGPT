import {
  getGetV2GetBuilderSuggestionsQueryKey,
  useGetV2GetBuilderSuggestions,
} from "@/app/api/__generated__/endpoints/default/default";
import { SuggestionsResponse } from "@/app/api/__generated__/models/suggestionsResponse";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";

export function useSuggestionContent() {
  const tenantScope = useBuilderTenantScope();
  const { data, isLoading, isError, error, refetch } =
    useGetV2GetBuilderSuggestions({
      request: getTenantRequestInit(
        tenantScope.organizationId,
        tenantScope.teamId,
        tenantScope.isReady,
      ),
      query: {
        enabled: tenantScope.isReady,
        queryKey: getTeamScopedQueryKey(
          getGetV2GetBuilderSuggestionsQueryKey(),
          tenantScope.organizationId,
          tenantScope.teamId,
        ),
        select: (x) => {
          return {
            suggestions: x.data as SuggestionsResponse,
            status: x.status,
          };
        },
      },
    });

  return { data, isLoading, isError, error, refetch };
}
