"use client";

import * as React from "react";
import { useRouter } from "next/navigation";

import { MagnifyingGlassIcon } from "@radix-ui/react-icons";

interface SearchBarProps {
  placeholder?: string;
  backgroundColor?: string;
  iconColor?: string;
  textColor?: string;
  placeholderColor?: string;
}

/** SearchBar component for user input and search functionality. */
export const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = 'Search for tasks like "optimise SEO"',
  backgroundColor = "bg-neutral-100",
  iconColor = "text-[#646464]",
  textColor = "text-[#707070]",
  placeholderColor = "text-[#707070]",
}) => {
  const router = useRouter();

  const [searchQuery, setSearchQuery] = React.useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    console.log(searchQuery);

    if (searchQuery.trim()) {
      // Encode the search term and navigate to the desired path
      const encodedTerm = encodeURIComponent(searchQuery);
      router.push(`/store/search?searchTerm=${encodedTerm}`);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="w-9/10 lg:w-[56.25rem]"
      data-testid="store-search-bar"
    >
      <div
        className={`h-12 px-4 py-2 md:h-[4.5rem] md:px-6 md:py-1 ${backgroundColor} flex items-center gap-2 rounded-full md:gap-5`}
      >
        <MagnifyingGlassIcon className={`h-5 w-5 md:h-7 md:w-7 ${iconColor}`} />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={placeholder}
          className={`flex-grow border-none bg-transparent ${textColor} font-['Geist'] text-lg font-normal leading-[2.25rem] tracking-tight md:text-xl placeholder:${placeholderColor} focus:outline-none`}
          data-testid="store-search-input"
        />
      </div>
    </form>
  );
};
