"use client";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { ArrowDownNarrowWideIcon } from "lucide-react";
import { useLibrarySortMenu } from "./useLibrarySortMenu";

interface Props {
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function LibrarySortMenu({ setLibrarySort }: Props) {
  const { handleSortChange } = useLibrarySortMenu({ setLibrarySort });
  return (
    <div className="flex items-center" data-testid="sort-by-dropdown">
      <span className="hidden whitespace-nowrap sm:inline">sort by</span>
      <Select onValueChange={handleSortChange}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-base underline underline-offset-4 shadow-none">
          <ArrowDownNarrowWideIcon className="h-4 w-4 sm:hidden" />
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
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
