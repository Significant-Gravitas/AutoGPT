"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import {
  ANY_STATUS,
  ANYONE,
  ownerChoices,
  QUERY_MAX,
  statusChoices,
  type TaskFilters,
} from "../taskFilters";
import { ChoiceMenu } from "@/app/(platform)/tasks/components/ChoiceMenu/ChoiceMenu";

interface Props {
  filters: TaskFilters;
  experts: Expert[];
  isFiltered: boolean;
  onQuery: (q: string) => void;
  onOwner: (owner: string) => void;
  onStatus: (status: string) => void;
  onReset: () => void;
}

/** Search the tasks by title and narrow them by who works them and their
 *  status; Reset clears all three. */
export function TaskFiltersBar({
  filters,
  experts,
  isFiltered,
  onQuery,
  onOwner,
  onStatus,
  onReset,
}: Props) {
  return (
    <div
      role="search"
      aria-label="Find tasks"
      className="flex min-w-0 flex-wrap items-center gap-2 px-1"
    >
      <SearchInput
        size="xsmall"
        aria-label="Search tasks"
        placeholder="Search tasks"
        value={filters.q}
        maxLength={QUERY_MAX}
        onChange={onQuery}
        className="max-w-64 min-w-40 flex-1"
      />
      <ChoiceMenu
        name="Owner"
        value={filters.owner}
        choices={ownerChoices(experts)}
        isActive={filters.owner !== ANYONE}
        onChange={onOwner}
      />
      <ChoiceMenu
        name="Status"
        value={filters.status}
        choices={statusChoices()}
        isActive={filters.status !== ANY_STATUS}
        onChange={onStatus}
      />
      {isFiltered ? (
        <Button
          type="button"
          variant="ghost-muted"
          size="xs"
          shape="pill"
          onClick={onReset}
        >
          Reset
        </Button>
      ) : null}
    </div>
  );
}
