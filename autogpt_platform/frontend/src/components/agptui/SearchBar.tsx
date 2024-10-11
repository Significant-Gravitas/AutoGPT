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
    <form onSubmit={handleSubmit} className="w-[900px]">
      <div
        className={`h-18 px-6 py-2.5 ${backgroundColor} flex items-center gap-5 rounded-full`}
      >
        <MagnifyingGlassIcon className={`h-7 w-7 ${iconColor}`} />
        <input
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          placeholder={placeholder}
          className={`flex-grow border-none bg-transparent ${textColor} font-neue text-2xl font-normal leading-9 tracking-tight placeholder:${placeholderColor} focus:outline-none`}
        />
      </div>
    </form>
  );
};
