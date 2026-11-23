"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import type { RefObject } from "react";
import {
  ANY_STATUS,
  ANYONE,
  DEFAULT_SORT,
  ownerChoices,
  QUERY_MAX,
  SORT_CHOICES,
  statusChoices,
  type TaskFilters,
} from "../taskFilters";
import { ChoiceMenu } from "@/app/(platform)/tasks/components/ChoiceMenu/ChoiceMenu";

interface Props {
  searchRef: RefObject<HTMLInputElement | null>;
  filters: TaskFilters;
  experts: Expert[];
  isFiltered: boolean;
  onQuery: (q: string) => void;
  onOwner: (owner: string) => void;
  onStatus: (status: string) => void;
  onSort: (sort: string) => void;
  onReset: () => void;
}

/** Search the tasks by title and narrow them by who works them (once
 *  there is more than Otto to pick) and their status; Reset clears all. */
export function TaskFiltersBar({
  searchRef,
  filters,
  experts,
  isFiltered,
  onQuery,
  onOwner,
  onStatus,
  onSort,
  onReset,
}: Props) {
  return (
    <div
      role="search"
      aria-label="Find tasks"
      className="flex min-w-0 flex-wrap items-center gap-2 px-1"
    >
      <SearchInput
        ref={searchRef}
        size="xsmall"
        aria-label="Search tasks"
        placeholder="Search tasks"
        value={filters.q}
        maxLength={QUERY_MAX}
        onChange={onQuery}
        className="max-w-64 min-w-40 flex-1"
      />
      {experts.length > 0 || filters.owner !== ANYONE ? (
        <ChoiceMenu
          name="Owner"
          value={filters.owner}
          choices={ownerChoices(experts)}
          isActive={filters.owner !== ANYONE}
          onChange={onOwner}
        />
      ) : null}
      <ChoiceMenu
        name="Status"
        value={filters.status}
        choices={statusChoices()}
        isActive={filters.status !== ANY_STATUS}
        onChange={onStatus}
      />
      <ChoiceMenu
        name="Sort"
        value={filters.sort}
        choices={SORT_CHOICES}
        isActive={filters.sort !== DEFAULT_SORT}
        onChange={onSort}
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
