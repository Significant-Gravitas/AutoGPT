import { Select } from "@/components/atoms/Select/Select";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { TeamFilter } from "../../helpers";

interface Props {
  query: string;
  onQueryChange: (next: string) => void;
  filter: TeamFilter;
  onFilterChange: (next: TeamFilter) => void;
}

const FILTER_OPTIONS = [
  { value: "all", label: "All experts" },
  { value: "scheduled", label: "Scheduled" },
  { value: "needs-setup", label: "Needs setup" },
  { value: "paused", label: "Paused" },
];

export function TeamRosterToolbar({
  query,
  onQueryChange,
  filter,
  onFilterChange,
}: Props) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      <SearchInput
        value={query}
        onChange={onQueryChange}
        size="small"
        placeholder="Search experts"
        aria-label="Search experts"
        className="w-full sm:w-56"
      />
      <Select
        id="team-filter"
        label="Filter"
        hideLabel
        size="small"
        value={filter}
        onValueChange={(next) => onFilterChange(next as TeamFilter)}
        options={FILTER_OPTIONS}
        // The hidden-label Select reserves `mb-6` for error text it never
        // shows, which pushes its trigger off the toolbar's centre line.
        wrapperClassName="w-40 !mb-0"
      />
    </div>
  );
}
