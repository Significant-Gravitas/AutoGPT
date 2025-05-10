"use client";

import * as React from "react";
import { useRouter } from "next/navigation";

import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { Input } from "../ui/input";
import { cn } from "@/lib/utils";

interface SearchBarProps {
  placeholder?: string;
  className?: string;
}

/** SearchBar component for user input and search functionality. */
export const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = 'Search for tasks like "optimise SEO"',
  className,
}) => {
  const router = useRouter();

  const [searchQuery, setSearchQuery] = React.useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (searchQuery.trim()) {
      // Encode the search term and navigate to the desired path
      const encodedTerm = encodeURIComponent(searchQuery);
      router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="store-search-bar"
      className={cn(
        `flex h-14 w-full items-center justify-center gap-2 rounded-full bg-[#F3F3F3] px-6 py-2.5 md:h-18 md:gap-5`,
        className,
      )}
    >
      <MagnifyingGlassIcon className={`h-5 w-5 text-[#020617] md:h-7 md:w-7`} />
      <Input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder={placeholder}
        className={`m-0 flex-grow border-none bg-transparent p-0 font-sans text-base font-normal text-zinc-800 shadow-none placeholder:text-neutral-500 focus:shadow-none focus:outline-none focus:ring-0 md:text-xl`}
        data-testid="store-search-input"
      />
    </form>
  );
};
