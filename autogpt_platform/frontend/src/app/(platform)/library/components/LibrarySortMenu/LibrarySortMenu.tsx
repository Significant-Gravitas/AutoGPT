"use client";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/atoms/Select/components/SelectPrimitives";
import { SortByDown02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useLibrarySortMenu } from "./useLibrarySortMenu";

interface Props {
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function LibrarySortMenu({ setLibrarySort }: Props) {
  const { handleSortChange } = useLibrarySortMenu({ setLibrarySort });
  return (
    <div className="flex items-center" data-testid="sort-by-dropdown">
      <span className="hidden text-sm whitespace-nowrap text-foreground-tertiary sm:inline">
        sort by
      </span>
      <Select onValueChange={handleSortChange}>
        <SelectTrigger
          // oxlint-disable-next-line design-system/no-restyle -- Select has no inline link-style trigger variant
          className="!m-0 ml-1 w-fit space-x-1 border-none !bg-transparent px-4 text-sm underline underline-offset-4 !shadow-none !ring-offset-transparent"
        >
          <Icon
            aria-hidden
            icon={SortByDown02Icon}
            className="h-4 w-4 sm:hidden"
          />
          <SelectValue placeholder="Last Modified" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value={LibraryAgentSort.createdAt}>
              Creation Date
            </SelectItem>
            <SelectItem value={LibraryAgentSort.updatedAt}>
              Last Modified
            </SelectItem>
            <SelectItem value={LibraryAgentSort.lastRunAt}>Last Run</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
