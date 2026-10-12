import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { FilterIconMenu } from "../../[expertId]/components/FilterIconMenu";
import { TeamFilter } from "../../helpers";

interface Props {
  query: string;
  onQueryChange: (next: string) => void;
  filter: TeamFilter;
  onFilterChange: (next: TeamFilter) => void;
}

const FILTER_OPTIONS: readonly { value: TeamFilter; label: string }[] = [
  { value: "all", label: "All experts" },
  { value: "scheduled", label: "Scheduled" },
  { value: "needs-setup", label: "Needs setup" },
  { value: "paused", label: "Paused" },
];

/** Search plus the same filter icon menu the expert page tabs use. */
export function TeamRosterToolbar({
  query,
  onQueryChange,
  filter,
  onFilterChange,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 sm:justify-end">
      <SearchInput
        value={query}
        onChange={onQueryChange}
        size="small"
        placeholder="Search experts"
        aria-label="Search experts"
        className="w-full sm:w-48"
      />
      <FilterIconMenu
        label="Filter experts"
        value={filter}
        options={FILTER_OPTIONS}
        defaultValue="all"
        onChange={onFilterChange}
      />
    </div>
  );
}
