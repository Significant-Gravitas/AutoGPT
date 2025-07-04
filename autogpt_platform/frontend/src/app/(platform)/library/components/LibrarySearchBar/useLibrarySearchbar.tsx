import { useRef, useState } from "react";
import { useLibraryPageContext } from "../state-provider";
import { debounce } from "lodash";

export const useLibrarySearchbar = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const { setSearchTerm } = useLibraryPageContext();

  const debouncedSearch = debounce((value: string) => {
    setSearchTerm(value);
  }, 300);

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchTerm = e.target.value;
    debouncedSearch(searchTerm);
  };

  const handleClear = (e: React.MouseEvent) => {
    if (inputRef.current) {
      inputRef.current.value = "";
      inputRef.current.blur();
      setSearchTerm("");
      e.preventDefault();
    }
    setIsFocused(false);
  };

  return {
    handleClear,
    handleSearchInput,
    isFocused,
    inputRef,
    setIsFocused,
  };
};
