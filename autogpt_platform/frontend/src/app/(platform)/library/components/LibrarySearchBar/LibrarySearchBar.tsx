"use client";

import { Input } from "@/components/atoms/Input/Input";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";
import { useLibrarySearchbar } from "./useLibrarySearchbar";

export default function LibrarySearchBar(): React.ReactNode {
  const { handleSearchInput } = useLibrarySearchbar();

  return (
    <div
      data-testid="search-bar"
      className="relative z-[21] -mb-6 flex items-center"
    >
      <MagnifyingGlassIcon
        width={18}
        height={18}
        className="absolute left-4 top-[34%] z-20 -translate-y-1/2 text-zinc-800"
      />

      <Input
        label="Search agents"
        id="library-search-bar"
        hideLabel
        onChange={handleSearchInput}
        className="min-w-[18rem] pl-12 lg:min-w-[30rem]"
        type="text"
        data-testid="library-textbox"
        placeholder="Search agents"
      />
    </div>
  );
}
