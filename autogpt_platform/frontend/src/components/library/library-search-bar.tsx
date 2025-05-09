"use client";
import { useRef, useState } from "react";
import debounce from "lodash/debounce";
import { Input } from "@/components/ui/input";
import { Search, X } from "lucide-react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useLibraryPageContext } from "@/app/(platform)/library/state-provider";

export default function LibrarySearchBar(): React.ReactNode {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const api = useBackendAPI();
  const { setAgentLoading, setAgents, librarySort, setSearchTerm } =
    useLibraryPageContext();

  const debouncedSearch = debounce(async (value: string) => {
    try {
      setAgentLoading(true);
      setSearchTerm(value);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const response = await api.listLibraryAgents({
        search_term: value,
        sort_by: librarySort,
        page: 1,
      });
      setAgents(response.agents);
      setAgentLoading(false);
    } catch (error) {
      console.error("Search failed:", error);
    }
  }, 300);
  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchTerm = e.target.value;
    debouncedSearch(searchTerm);
  };

  return (
    <div
      onClick={() => inputRef.current?.focus()}
      className="relative z-[21] mx-auto flex h-[50px] w-full max-w-[500px] flex-1 cursor-pointer items-center rounded-[45px] bg-[#EDEDED] px-[24px] py-[10px]"
    >
      <Search
        className="mr-2 h-[29px] w-[29px] text-neutral-900"
        strokeWidth={1.25}
      />

      <Input
        ref={inputRef}
        onFocus={() => setIsFocused(true)}
        onBlur={() => !inputRef.current?.value && setIsFocused(false)}
        onChange={handleSearchInput}
        className="flex-1 border-none font-sans text-[16px] font-normal leading-7 shadow-none focus:shadow-none focus:ring-0"
        type="text"
        placeholder="Search agents"
      />

      {isFocused && inputRef.current?.value && (
        <X
          className="ml-2 h-[29px] w-[29px] cursor-pointer text-neutral-900"
          strokeWidth={1.25}
          onClick={(e: React.MouseEvent) => {
            if (inputRef.current) {
              debouncedSearch("");
              inputRef.current.value = "";
              inputRef.current.blur();
              e.preventDefault();
            }
            setIsFocused(false);
          }}
        />
      )}
    </div>
  );
}
