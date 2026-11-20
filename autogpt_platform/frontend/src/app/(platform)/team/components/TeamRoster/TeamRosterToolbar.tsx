import { TeamFilter } from "../../helpers";
import { ToolbarSearch } from "../ToolbarSearch";
import { TeamFilterMenu } from "./TeamFilterMenu";

interface Props {
  query: string;
  onQueryChange: (next: string) => void;
  filter: TeamFilter;
  onFilterChange: (next: TeamFilter) => void;
}

/** Search and filter for the roster: the Files page's rounded Input and a
 *  pill filter menu. */
export function TeamRosterToolbar({
  query,
  onQueryChange,
  filter,
  onFilterChange,
}: Props) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
      <ToolbarSearch
        id="team-roster-search"
        label="Search experts"
        value={query}
        onChange={onQueryChange}
        className="sm:w-72"
      />
      <TeamFilterMenu value={filter} onChange={onFilterChange} />
    </div>
  );
}
