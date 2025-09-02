"use client";
import { ArrowDownNarrowWideIcon } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LibraryAgentSortEnum as LibraryAgentSort } from "@/lib/autogpt-server-api/types";
import { useLibrarySortMenu } from "./useLibrarySortMenu";

export default function LibrarySortMenu(): React.ReactNode {
  const { handleSortChange } = useLibrarySortMenu();
  return (
    <div className="flex items-center" data-testid="sort-by-dropdown">
      <span className="hidden whitespace-nowrap sm:inline">sort by</span>
      <Select onValueChange={handleSortChange}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-base underline underline-offset-4 shadow-none">
          <ArrowDownNarrowWideIcon className="h-4 w-4 sm:hidden" />
          <SelectValue placeholder="Favorites First" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value={LibraryAgentSort.FAVORITES_FIRST}>
              Favorites First
            </SelectItem>
            <SelectItem value={LibraryAgentSort.CREATED_AT}>
              Creation Date
            </SelectItem>
            <SelectItem value={LibraryAgentSort.UPDATED_AT}>
              Last Modified
            </SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
