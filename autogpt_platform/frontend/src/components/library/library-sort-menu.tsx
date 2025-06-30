import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { LibraryAgentSortEnum } from "@/lib/autogpt-server-api/types";
import { useLibraryPageContext } from "@/app/(platform)/library/state-provider";
import { ArrowDownNarrowWideIcon } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMemo } from "react";

const SORT_LABELS: Record<LibraryAgentSortEnum, string> = {
  [LibraryAgentSortEnum.CREATED_AT]: "Creation Date",
  [LibraryAgentSortEnum.UPDATED_AT]: "Last Modified",
  [LibraryAgentSortEnum.LAST_EXECUTION]: "Last Run",
};

export default function LibrarySortMenu(): React.ReactNode {
  const api = useBackendAPI();
  const { setAgentLoading, setAgents, setLibrarySort, searchTerm, librarySort } =
    useLibraryPageContext();

  const handleSortChange = async (value: LibraryAgentSortEnum) => {
    setLibrarySort(value);
    setAgentLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const response = await api.listLibraryAgents({
      search_term: searchTerm,
      sort_by: value,
      page: 1,
    });
    setAgents(response.agents);
    setAgentLoading(false);
  };

  const placeholderText = useMemo(
    () => SORT_LABELS[librarySort] || SORT_LABELS[LibraryAgentSortEnum.UPDATED_AT],
    [librarySort]
  );

  return (
    <div className="flex items-center">
      <span className="hidden whitespace-nowrap sm:inline">sort by</span>
      <Select onValueChange={handleSortChange} defaultValue={librarySort}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-base underline underline-offset-4 shadow-none">
          <ArrowDownNarrowWideIcon className="h-4 w-4 sm:hidden" />
          <SelectValue placeholder={placeholderText} />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {Object.entries(SORT_LABELS).map(([value, label]) => (
              <SelectItem key={value} value={value}>
                {label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
