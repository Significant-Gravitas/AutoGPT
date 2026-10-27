"use client";
import { useSearchbar } from "./useSearchBar";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface SearchBarProps {
  placeholder?: string;
  width?: string;
  height?: string;
  defaultValue?: string;
  onSubmit?: (query: string) => void;
}

export function SearchBar({
  placeholder = 'Search for tasks like "optimize SEO"',
  width = "w-full lg:w-[56.25rem]",
  height = "h-[3.8rem]",
  defaultValue,
  onSubmit,
}: SearchBarProps) {
  const { handleSubmit, setSearchQuery, searchQuery } = useSearchbar({
    defaultValue,
    onSubmit,
  });

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="store-search-bar"
      // TODO(tokens): no fitting role for focus-within:border-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour); no fitting role for focus-within:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
      className={`${width} ${height} flex items-center gap-3 rounded-full border border-border-base bg-surface px-4 shadow-none focus-within:border-zinc-400 focus-within:ring-1 focus-within:ring-zinc-400 focus-within:ring-offset-0`}
    >
      <Icon
        icon={Search01Icon}
        size={20}
        className="text-foreground-subtle md:h-6 md:w-6"
        aria-hidden="true"
      />
      <input
        type="text"
        aria-label={placeholder}
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder={placeholder}
        className="flex-grow border-none bg-transparent text-base font-normal text-foreground-base placeholder:text-base placeholder:font-normal placeholder:text-foreground-subtle focus:outline-none md:text-lg md:placeholder:text-lg"
        data-testid="store-search-input"
      />
    </form>
  );
}
