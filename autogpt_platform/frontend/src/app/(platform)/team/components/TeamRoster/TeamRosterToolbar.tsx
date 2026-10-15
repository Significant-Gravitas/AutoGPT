import {
  Alert01Icon,
  Calendar03Icon,
  PauseIcon,
  Search01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { ChangeEvent } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { TeamFilter } from "../../helpers";

interface Props {
  query: string;
  onQueryChange: (next: string) => void;
  filter: TeamFilter;
  onFilterChange: (next: TeamFilter) => void;
}

const FILTER_OPTIONS = [
  { value: "all", label: "All experts", icon: UserGroupIcon },
  { value: "scheduled", label: "Scheduled", icon: Calendar03Icon },
  { value: "needs-setup", label: "Needs setup", icon: Alert01Icon },
  { value: "paused", label: "Paused", icon: PauseIcon },
] as const;

/** Search and filter for the roster, built from the same Input and Select
 *  atoms as the Files page toolbar. */
export function TeamRosterToolbar({
  query,
  onQueryChange,
  filter,
  onFilterChange,
}: Props) {
  function handleQueryChange(event: ChangeEvent<HTMLInputElement>) {
    onQueryChange(event.target.value);
  }

  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
      <div className="relative flex w-full items-center sm:w-72">
        <Icon
          icon={Search01Icon}
          size={16}
          className="pointer-events-none absolute left-4 top-1/2 z-20 -translate-y-1/2 text-zinc-500"
        />
        <Input
          id="team-roster-search"
          label="Search experts"
          hideLabel
          type="text"
          size="small"
          value={query}
          onChange={handleQueryChange}
          placeholder="Search experts"
          className="rounded-full pl-10"
          wrapperClassName="!mb-0 w-full"
        />
      </div>
      <Select
        id="team-roster-filter"
        label="Filter experts"
        hideLabel
        size="small"
        value={filter}
        onValueChange={(next) => onFilterChange(next as TeamFilter)}
        options={FILTER_OPTIONS.map((option) => ({
          value: option.value,
          label: option.label,
          icon: <Icon icon={option.icon} size={14} className="text-zinc-500" />,
        }))}
        className="rounded-full sm:w-44"
        wrapperClassName="!mb-0"
      />
    </div>
  );
}
