import debounce from "lodash/debounce";
import { useCallback, useEffect, useRef, useState } from "react";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { getGetV2GetBuilderSuggestionsQueryKey } from "@/app/api/__generated__/endpoints/default/default";

const SEARCH_DEBOUNCE_MS = 300;

export const useBlockMenuSearchBar = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [localQuery, setLocalQuery] = useState("");
  const { setSearchQuery, setSearchId, searchQuery } = useBlockMenuStore();
  const queryClient = getQueryClient();

  const clearSearchSession = useCallback(() => {
    setSearchId(undefined);
    queryClient.invalidateQueries({
      queryKey: getGetV2GetBuilderSuggestionsQueryKey(),
    });
  }, [queryClient, setSearchId]);

  const debouncedSetSearchQuery = useCallback(
    debounce((value: string) => {
      setSearchQuery(value);
      if (value.length === 0) {
        clearSearchSession();
      }
    }, SEARCH_DEBOUNCE_MS),
    [clearSearchSession, setSearchQuery],
  );

  useEffect(() => {
    return () => {
      debouncedSetSearchQuery.cancel();
    };
  }, [debouncedSetSearchQuery]);

  const handleClear = () => {
    setLocalQuery("");
    setSearchQuery("");
    clearSearchSession();
    debouncedSetSearchQuery.cancel();
  };

  useEffect(() => {
    setLocalQuery(searchQuery);
  }, [searchQuery]);

  return {
    handleClear,
    inputRef,
    localQuery,
    setLocalQuery,
    debouncedSetSearchQuery,
  };
};
