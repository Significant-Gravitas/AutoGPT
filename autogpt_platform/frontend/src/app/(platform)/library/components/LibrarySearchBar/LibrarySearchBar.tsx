"use client";

import { Input } from "@/components/atoms/Input/Input";
import { useLibrarySearchbar } from "./useLibrarySearchbar";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  setSearchTerm: (value: string) => void;
}

export function LibrarySearchBar({ setSearchTerm }: Props) {
  const { handleSearchInput } = useLibrarySearchbar({ setSearchTerm });

  return (
    <div
      data-testid="search-bar"
      className="relative z-[21] -mb-6 flex w-full items-center md:w-auto"
    >
      <span className="contents text-foreground-emphasis">
        <Icon
          icon={Search01Icon}
          width={18}
          height={18}
          className="absolute left-4 top-[34%] z-20 -translate-y-1/2"
        />
      </span>

      <Input
        label="Search workflows"
        id="library-search-bar"
        hideLabel
        onChange={handleSearchInput}
        // oxlint-disable-next-line design-system/no-restyle -- Input has no leading-icon slot; pl-12 clears the search icon
        className="min-w-[18rem] pl-12 lg:min-w-[30rem]"
        type="text"
        data-testid="library-textbox"
        placeholder="Search workflows"
      />
    </div>
  );
}
