import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Select } from "@/components/atoms/Select/Select";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { cn } from "@/lib/utils";
import { Grid2X2Icon, ListViewIcon } from "@hugeicons/core-free-icons";
import { TeamFilter } from "../../helpers";
import { TeamView } from "./useTeamRosterView";

interface Props {
  query: string;
  onQueryChange: (next: string) => void;
  filter: TeamFilter;
  onFilterChange: (next: TeamFilter) => void;
  view: TeamView;
  onViewChange: (next: TeamView) => void;
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
  view,
  onViewChange,
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
      <div className="flex h-9 items-center gap-1 rounded-xl border border-zinc-200 px-0.5">
        <ViewToggle
          label="Card view"
          icon={Grid2X2Icon}
          active={view === "cards"}
          onClick={() => onViewChange("cards")}
        />
        <ViewToggle
          label="Table view"
          icon={ListViewIcon}
          active={view === "table"}
          onClick={() => onViewChange("table")}
        />
      </div>
    </div>
  );
}

interface ViewToggleProps {
  label: string;
  icon: React.ComponentProps<typeof Icon>["icon"];
  active: boolean;
  onClick: () => void;
}

function ViewToggle({ label, icon, active, onClick }: ViewToggleProps) {
  return (
    <Button
      variant="ghost"
      size="small"
      aria-label={label}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "!h-8 !min-w-0 !rounded-lg px-2",
        active && "bg-zinc-100 hover:bg-zinc-100",
      )}
    >
      <Icon icon={icon} size={16} />
    </Button>
  );
}
