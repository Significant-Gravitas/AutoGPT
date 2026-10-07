"use client";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowDownNarrowWideIcon } from "@hugeicons/core-free-icons";
import { useLibrarySortMenu } from "./useLibrarySortMenu";

interface Props {
  setLibrarySort: (value: LibraryAgentSort) => void;
}

const SORT_OPTIONS = [
  { value: LibraryAgentSort.createdAt, label: "Creation Date" },
  { value: LibraryAgentSort.updatedAt, label: "Last Modified" },
  { value: LibraryAgentSort.lastRunAt, label: "Last Run" },
];

export function LibrarySortMenu({ setLibrarySort }: Props) {
  const { handleSortChange } = useLibrarySortMenu({ setLibrarySort });
  return (
    <div className="flex items-center" data-testid="sort-by-dropdown">
      <Text
        variant="body"
        as="span"
        tone="muted"
        className="hidden whitespace-nowrap sm:inline"
      >
        sort by
      </Text>
      <Icon
        icon={ArrowDownNarrowWideIcon}
        size={16}
        className="ml-1 sm:hidden"
      />
      <Select
        id="library-sort"
        label="Sort agents"
        hideLabel
        placeholder="Last Modified"
        onValueChange={(value) => handleSortChange(value as LibraryAgentSort)}
        options={SORT_OPTIONS}
        size="small"
        className="ml-1 w-fit border-none bg-transparent text-sm underline underline-offset-4 shadow-none [&[data-placeholder]>span]:text-black"
        wrapperClassName="mb-0"
      />
    </div>
  );
}
