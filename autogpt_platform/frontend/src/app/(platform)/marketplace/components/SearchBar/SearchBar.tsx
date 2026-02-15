"use client";

import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useSearchbar } from "./useSearchBar";

interface SearchBarProps {
  placeholder?: string;
  backgroundColor?: string;
  iconColor?: string;
  textColor?: string;
  placeholderColor?: string;
  width?: string;
  height?: string;
}

export const SearchBar = ({
  placeholder = 'Search for tasks like "optimise SEO"',
  backgroundColor = "bg-neutral-100 dark:bg-neutral-800",
  iconColor = "text-neutral-500 dark:text-neutral-400",
  textColor = "text-neutral-500 dark:text-neutral-200",
  placeholderColor = "text-neutral-500 dark:text-neutral-400",
  width = "w-9/10 lg:w-[56.25rem]",
  height = "h-[60px]",
}: SearchBarProps) => {
  const { handleSubmit, setSearchQuery, searchQuery } = useSearchbar();

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="store-search-bar"
      className={`${width} ${height} px-4 pt-2 md:px-6 md:pt-1 ${backgroundColor} flex items-center justify-center gap-2 rounded-full md:gap-5`}
    >
      <MagnifyingGlassIcon className={`h-5 w-5 md:h-7 md:w-7 ${iconColor}`} />
      <input
        type="search"
        name="search"
        autoComplete="off"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder={placeholder}
        aria-label="Search for AI agents"
        className={`flex-grow border-none bg-transparent ${textColor} font-sans text-lg font-normal leading-[2.25rem] tracking-tight md:text-xl placeholder:${placeholderColor} focus:outline-none`}
        data-testid="store-search-input"
      />
    </form>
  );
};
