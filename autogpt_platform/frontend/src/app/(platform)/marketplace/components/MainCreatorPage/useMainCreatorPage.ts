import {
  useGetV2GetCreatorDetails,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { MarketplaceCreatorPageParams } from "../../creator/[creator]/page";
import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";

interface useMainCreatorPageProps {
  params: MarketplaceCreatorPageParams;
}

export const useMainCreatorPage = ({ params }: useMainCreatorPageProps) => {
  const {
    data: creatorAgents,
    isLoading: isCreatorAgentsLoading,
    isError: isCreatorAgentsError,
  } = useGetV2ListStoreAgents(
    { creator: params.creator },
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const {
    data: creator,
    isLoading: isCreatorDetailsLoading,
    isError: isCreatorDetailsError,
  } = useGetV2GetCreatorDetails(params.creator, {
    query: {
      select: (x) => {
        return x.data as CreatorDetails;
      },
    },
  });

  const isLoading = isCreatorAgentsLoading || isCreatorDetailsLoading;
  const hasError = isCreatorAgentsError || isCreatorDetailsError;

  return {
    creatorAgents,
    creator,
    isLoading,
    hasError,
  };
};
