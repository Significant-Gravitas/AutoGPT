import { debounce } from "lodash";
import { useCallback, useEffect } from "react";
import { useLibraryPageContext } from "../state-provider";

export function useLibrarySearchbar() {
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

  function handleSearchInput(e: React.ChangeEvent<HTMLInputElement>) {
    const searchTerm = e.target.value;
    debouncedSearch(searchTerm);
  }

  return {
    handleSearchInput,
  };
}
