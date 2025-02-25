import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { LibraryAgentSortEnum } from "@/lib/autogpt-server-api/types";
import { useLibraryPageContext } from "@/app/library/state-provider";
import { ArrowDownNarrowWideIcon } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function LibrarySortMenu(): React.ReactNode {
  const api = useBackendAPI();
  const { setAgentLoading, setAgents, setLibrarySort, searchTerm } =
    useLibraryPageContext();
  const handleSortChange = async (value: LibraryAgentSortEnum) => {
    setLibrarySort(value);
    setAgentLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    let response = await api.listLibraryAgents({
      search_term: searchTerm,
      sort_by: value,
      page: 1,
    });
    setAgents(response.agents);
    setAgentLoading(false);
  };

  return (
    <div className="flex items-center">
      <span className="hidden whitespace-nowrap sm:inline">sort by</span>
      <Select onValueChange={handleSortChange}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-base underline underline-offset-4 shadow-none">
          <ArrowDownNarrowWideIcon className="h-4 w-4 sm:hidden" />
          <SelectValue placeholder="Last Modified" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value={LibraryAgentSortEnum.CREATED_AT}>
              Creation Date
            </SelectItem>
            <SelectItem value={LibraryAgentSortEnum.UPDATED_AT}>
              Last Modified
            </SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
