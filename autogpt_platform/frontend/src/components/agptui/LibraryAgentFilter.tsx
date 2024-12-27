import { GraphMeta } from "@/lib/autogpt-server-api";
import { Dispatch, SetStateAction } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Filter } from "lucide-react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

type SortValue =
  | "most_recent"
  | "highest_runtime"
  | "most_runs"
  | "alphabetical"
  | "last_modified";

const LibraryAgentFilter = ({
  setAgents,
  setAgentLoading,
}: {
  setAgents: Dispatch<SetStateAction<GraphMeta[]>>;
  setAgentLoading: Dispatch<SetStateAction<boolean>>;
}) => {
  const api = useBackendAPI();
  const handleSortChange = async (value: SortValue) => {
    setAgentLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    let response = await api.librarySearchAgent("", undefined, value);
    setAgents(response.agents);
    setAgentLoading(false);
  };

  return (
    <div className="flex items-center">
      <span className="hidden sm:inline">sort by</span>
      <Select onValueChange={handleSortChange}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none pl-2 shadow-md">
          <Filter className="h-4 w-4 sm:hidden" />
          <SelectValue
            placeholder="most Recent"
            className={
              "font-sans text-[14px] font-[500] leading-[24px] text-neutral-600"
            }
          />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup
            className={
              "font-sans text-[14px] font-[500] leading-[24px] text-neutral-600"
            }
          >
            <SelectItem value="most_recent">Most Recent</SelectItem>
            <SelectItem value="highest_runtime">Highest Runtime</SelectItem>
            <SelectItem value="most_runs">Most Runs</SelectItem>
            <SelectItem value="alphabetical">Alphabetical</SelectItem>
            <SelectItem value="last_modified">Last Modified</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
};

export default LibraryAgentFilter;
