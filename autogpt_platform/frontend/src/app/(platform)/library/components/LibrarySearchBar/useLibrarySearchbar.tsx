import { debounce } from "lodash";
import { useLibraryPageContext } from "../state-provider";

export const useLibrarySearchbar = () => {
  const { setSearchTerm } = useLibraryPageContext();

  const debouncedSearch = debounce((value: string) => {
    setSearchTerm(value);
  }, 300);

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchTerm = e.target.value;
    debouncedSearch(searchTerm);
  };

  return {
    handleSearchInput,
  };
};
