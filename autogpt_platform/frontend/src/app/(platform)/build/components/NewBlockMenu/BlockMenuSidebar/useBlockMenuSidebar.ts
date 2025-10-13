import { useGetV2GetBuilderItemCounts } from "@/app/api/__generated__/endpoints/default/default";
import { CountResponse } from "@/app/api/__generated__/models/countResponse";
import { useBlockMenuStore } from "../../../stores/blockMenuStore";

export const useBlockMenuSidebar = () => {
  const { defaultState, setDefaultState } = useBlockMenuStore();

  const { data, isLoading, isError, error } = useGetV2GetBuilderItemCounts({
    query: {
      select: (x) => {
        return {
          blockCounts: x.data as CountResponse,
          status: x.status,
        };
      },
    },
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
