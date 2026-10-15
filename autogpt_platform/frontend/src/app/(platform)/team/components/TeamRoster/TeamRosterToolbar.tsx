import { Search01Icon } from "@hugeicons/core-free-icons";
import type { ChangeEvent } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { TeamFilter } from "../../helpers";
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
      <TeamFilterMenu value={filter} onChange={onFilterChange} />
    </div>
  );
}
