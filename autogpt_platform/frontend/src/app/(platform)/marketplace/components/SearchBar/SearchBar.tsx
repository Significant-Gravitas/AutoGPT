"use client";

import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useSearchbar } from "./useSearchBar";

interface SearchBarProps {
  placeholder?: string;
  width?: string;
  height?: string;
}

export const SearchBar = ({
  placeholder = 'Search for tasks like "optimise SEO"',
  width = "w-9/10 lg:w-[56.25rem]",
  height = "h-[2.875rem]",
}: SearchBarProps) => {
  const { handleSubmit, setSearchQuery, searchQuery } = useSearchbar();

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="store-search-bar"
      className={`${width} ${height} flex items-center gap-3 rounded-3xl border border-zinc-200 bg-white px-4 shadow-none focus-within:border-zinc-400 focus-within:ring-1 focus-within:ring-zinc-400 focus-within:ring-offset-0`}
    >
      <MagnifyingGlassIcon className="h-5 w-5 text-zinc-400" />
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder={placeholder}
        className="flex-grow border-none bg-transparent font-normal text-black placeholder:font-normal placeholder:text-zinc-400 focus:outline-none"
        data-testid="store-search-input"
      />
    </form>
  );
};
