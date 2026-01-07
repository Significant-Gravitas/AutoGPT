import { debounce } from "lodash";
import { useCallback, useEffect } from "react";
import { useLibraryPageContext } from "../state-provider";

export const useLibrarySearchbar = () => {
  const { setSearchTerm } = useLibraryPageContext();

  const debouncedSearch = useCallback(
    debounce((value: string) => {
      setSearchTerm(value);
    }, 300),
    [setSearchTerm],
  );

  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
    };
  }, [debouncedSearch]);

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchTerm = e.target.value;
    debouncedSearch(searchTerm);
  };

  return {
    handleSearchInput,
  };
};
