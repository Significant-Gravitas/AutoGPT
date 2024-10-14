import * as React from "react";

import { MagnifyingGlassIcon } from "@radix-ui/react-icons";

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  backgroundColor?: string;
  iconColor?: string;
  textColor?: string;
  placeholderColor?: string;
}

/** SearchBar component for user input and search functionality. */
export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  placeholder = 'Search for tasks like "optimise SEO"',
  backgroundColor = "bg-neutral-100",
  iconColor = "text-[#646464]",
  textColor = "text-[#878787]",
  placeholderColor = "text-[#878787]",
}) => {
  const [searchQuery, setSearchQuery] = React.useState("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSearch(searchQuery);
  };

  return (
    <form onSubmit={handleSubmit} className="w-screen lg:w-[56.25rem]">
      <div
        className={`h-12 md:h-[4.5rem] px-4 py-2md:px-6 md:py-[0.625rem] ${backgroundColor} flex items-center gap-2 md:gap-5 rounded-full`}
      >
        <MagnifyingGlassIcon className={`h-5 w-5 md:h-7 md:w-7 ${iconColor}`} />
        <input
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          placeholder={placeholder}
          className={`flex-grow border-none bg-transparent ${textColor} font-neue text-lg md:text-xl font-normal leading-[2.25rem] tracking-tight placeholder:${placeholderColor} focus:outline-none`}
        />
      </div>
    </form>
  );
};
