"use client";

import { MagnifyingGlass } from "@phosphor-icons/react";
import { useSearchbar } from "./useSearchBar";

interface SearchBarProps {
  placeholder?: string;
  width?: string;
  height?: string;
}

export function SearchBar({
  placeholder = 'Search for tasks like "optimise SEO"',
  width = "w-full lg:w-[56.25rem]",
  height = "h-[3.8rem]",
}: SearchBarProps) {
  const { handleSubmit, setSearchQuery, searchQuery } = useSearchbar();

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="store-search-bar"
      className={`${width} ${height} flex items-center gap-3 rounded-full border border-zinc-200 bg-white px-4 shadow-none focus-within:border-zinc-400 focus-within:ring-1 focus-within:ring-zinc-400 focus-within:ring-offset-0`}
    >
      <MagnifyingGlass
        size={20}
        className="text-zinc-400 md:h-6 md:w-6"
        aria-hidden="true"
      />
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder={placeholder}
        className="flex-grow border-none bg-transparent text-base font-normal text-black placeholder:text-base placeholder:font-normal placeholder:text-zinc-400 focus:outline-none md:text-lg md:placeholder:text-lg"
        data-testid="store-search-input"
      />
    </form>
  );
}
