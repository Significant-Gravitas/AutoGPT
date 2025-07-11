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
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { useLibrarySortMenu } from "./useLibrarySortMenu";

const defaultSort = LibraryAgentSort.lastExecuted;

export default function LibrarySortMenu(): React.ReactNode {
  const { getSortLabel, setLibrarySort } = useLibrarySortMenu();

  return (
    <div className="flex items-center">
      <span className="hidden whitespace-nowrap sm:inline">sort by</span>
      <Select
        onValueChange={(value) => setLibrarySort(value as LibraryAgentSort)}
        defaultValue={defaultSort}
      >
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-base underline underline-offset-4 shadow-none">
          <ArrowDownNarrowWideIcon className="h-4 w-4 sm:hidden" />
          <SelectValue placeholder={getSortLabel(defaultSort)} />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {Object.entries(LibraryAgentSort).map(([key, value]) => (
              <SelectItem key={key} value={value}>
                {getSortLabel(value)}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
