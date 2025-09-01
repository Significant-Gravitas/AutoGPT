import { useGetV2GetBuilderItemCounts } from "@/app/api/__generated__/endpoints/default/default";
import { useBlockMenuContext } from "../block-menu-provider";
import { CountResponse } from "@/app/api/__generated__/models/countResponse";

export const useBlockMenuSidebar = () => {
  const { defaultState, setDefaultState } = useBlockMenuContext();

  const { data: blockCounts, isLoading, isError, error} = useGetV2GetBuilderItemCounts({
    query : {
        select : (x) =>{
            return x.data as CountResponse
        }
    }
  });

  return {
    blockCounts,
    setDefaultState,
    defaultState,
    isLoading,
    isError,
    error,
  }
};