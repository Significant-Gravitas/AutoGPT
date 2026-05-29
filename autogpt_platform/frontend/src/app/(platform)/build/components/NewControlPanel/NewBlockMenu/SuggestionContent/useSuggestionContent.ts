import { useGetV2GetBuilderSuggestions } from "@/app/api/__generated__/endpoints/default/default";
import { SuggestionsResponse } from "@/app/api/__generated__/models/suggestionsResponse";

export const useSuggestionContent = () => {
  const { data, isLoading, isError, error, refetch } =
    useGetV2GetBuilderSuggestions({
      query: {
        select: (x) => {
          return {
            suggestions: x.data as SuggestionsResponse,
            status: x.status,
          };
        },
      },
    });

  return { data, isLoading, isError, error, refetch };
};
