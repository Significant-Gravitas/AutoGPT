import { useGetV2GetBuilderSuggestions } from "@/app/api/__generated__/endpoints/default/default";
import { SuggestionsResponse } from "@/app/api/__generated__/models/suggestionsResponse";

export const useSuggestionContent = () => {
    const { data : suggestions, isLoading , isError, error, refetch } = useGetV2GetBuilderSuggestions({
        query: {
            select: (x) => {
                return x.data as SuggestionsResponse;
            }
        }
    });

    return { suggestions, isLoading, isError, error, refetch };
};