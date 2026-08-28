import {
  getGetV2GetBuilderItemCountsQueryKey,
  useGetV2GetBuilderItemCounts,
} from "@/app/api/__generated__/endpoints/default/default";
import { CountResponse } from "@/app/api/__generated__/models/countResponse";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

export const useBlockMenuSidebar = () => {
  const { defaultState, setDefaultState } = useBlockMenuStore();
  const tenantScope = useBuilderTenantScope();

  const { data, isLoading, isError, error } = useGetV2GetBuilderItemCounts({
    query: {
      enabled: tenantScope.isReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2GetBuilderItemCountsQueryKey(),
        tenantScope.organizationId,
        tenantScope.teamId,
      ),
      select: (x) => {
        return {
          blockCounts: x.data as CountResponse,
          status: x.status,
        };
      },
    },
    request: getTenantRequestInit(
      tenantScope.organizationId,
      tenantScope.teamId,
      tenantScope.isReady,
    ),
  });

  return {
    data,
    setDefaultState,
    defaultState,
    isLoading,
    isError,
    error,
  };
};
