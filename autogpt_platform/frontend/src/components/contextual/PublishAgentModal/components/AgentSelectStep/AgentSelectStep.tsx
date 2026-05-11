"use client";

import {
  CaretLeftIcon,
  CaretRightIcon,
  CheckCircleIcon,
  FunnelIcon,
  PlusIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";

import { Text } from "../../../../atoms/Text/Text";
import { Button } from "../../../../atoms/Button/Button";
import { Select } from "../../../../atoms/Select/Select";
import { StepHeader } from "../StepHeader";
import { StepFooter } from "../StepFooter";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { useAgentSelectStep } from "./useAgentSelectStep";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { MyAgentsSortBy } from "@/app/api/__generated__/models/myAgentsSortBy";
import { MyAgentsStatusFilter } from "@/app/api/__generated__/models/myAgentsStatusFilter";

interface Props {
  onSelect: (agentId: string, agentVersion: number) => void;
  onCancel: () => void;
  onNext: (
    agentId: string,
    agentVersion: number,
    agentData: {
      name: string;
      description: string;
      imageSrc: string;
      recommendedScheduleCron: string | null;
    },
  ) => void;
  onOpenBuilder: () => void;
}

const SORT_OPTIONS = [
  { value: MyAgentsSortBy.most_recent, label: "Sort by: Most recent" },
  { value: MyAgentsSortBy.name, label: "Sort by: Alphabetical" },
];

const FILTER_OPTIONS: { value: MyAgentsStatusFilter; label: string }[] = [
  { value: MyAgentsStatusFilter.never_submitted, label: "Never submitted" },
  { value: MyAgentsStatusFilter.draft, label: "Draft" },
  { value: MyAgentsStatusFilter.submitted, label: "Submitted / In review" },
  { value: MyAgentsStatusFilter.published, label: "Published" },
];

export function AgentSelectStep({
  onSelect,
  onCancel,
  onNext,
  onOpenBuilder,
}: Props) {
  const {
    myAgents,
    isLoading,
    isFetching,
    error,
    selectedAgentId,
    page,
    totalPages,
    totalItems,
    pageSize,
    sortBy,
    statuses,
    hasNoResults,
    handleAgentClick,
    handleNext,
    handleSortChange,
    toggleStatus,
    clearStatuses,
    goToPage,
    isNextDisabled,
  } = useAgentSelectStep({ onSelect, onNext });

  if (error) {
    return (
      <div className="mx-auto flex w-full flex-col">
        <StepHeader
          title="Choose an agent"
          description="Pick the saved agent version you want to send to marketplace review."
          currentStep="select"
        />
        <div className="mt-5 flex min-h-[320px] flex-col items-center justify-center gap-4 rounded-[18px] border border-rose-100 bg-rose-50 px-6 py-8 text-center">
          <WarningCircleIcon
            size={32}
            weight="duotone"
            className="text-rose-600"
          />
          <Text variant="large-medium" className="text-rose-900">
            We could not load your agents
          </Text>
          <Text variant="body" className="max-w-[420px] text-rose-700">
            Refresh the list and try again. Your current marketplace submissions
            are unchanged.
          </Text>
          <Button onClick={() => window.location.reload()} variant="secondary">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const hasActiveFilters = statuses.length > 0;
  const showEmpty =
    !isLoading &&
    !hasActiveFilters &&
    totalItems === 0 &&
    myAgents.length === 0;

  return (
    <div className="mx-auto flex w-full flex-col">
      <StepHeader
        title="Choose an agent"
        description="Pick the saved agent version you want to send to marketplace review."
        currentStep="select"
      />

      {showEmpty ? (
        <div className="mt-5 flex min-h-[320px] flex-col items-center justify-center gap-4 rounded-[18px] border border-dashed border-zinc-300 bg-zinc-50 px-6 py-8 text-center">
          <div className="flex size-11 items-center justify-center rounded-full bg-white text-zinc-700 shadow-[0_1px_2px_rgba(15,15,20,0.06)]">
            <PlusIcon size={20} weight="bold" />
          </div>
          <Text variant="large-medium" className="text-textBlack">
            No publishable agents yet
          </Text>
          <Text variant="body" className="max-w-[460px] text-zinc-600">
            Create and save an agent in the builder. It will appear here when a
            version is ready to submit.
          </Text>
          <Button onClick={onOpenBuilder}>Open builder</Button>
        </div>
      ) : (
        <>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 pb-3">
            <div className="w-full sm:w-[220px]">
              <Select
                id="agent-sort"
                label="Sort agents"
                hideLabel
                size="small"
                value={sortBy}
                onValueChange={handleSortChange}
                options={SORT_OPTIONS}
              />
            </div>
            <Popover>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className={cn(
                    "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-offset-1",
                    hasActiveFilters
                      ? "border-purple-300 bg-purple-50 text-purple-700"
                      : "border-zinc-200 bg-white text-zinc-700 hover:border-zinc-300",
                  )}
                >
                  <FunnelIcon size={14} weight="bold" />
                  Filter
                  {hasActiveFilters ? (
                    <span className="inline-flex size-5 items-center justify-center rounded-full bg-purple-500 text-[11px] font-semibold text-white">
                      {statuses.length}
                    </span>
                  ) : null}
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-60 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <Text
                    variant="small-medium"
                    as="span"
                    className="text-textBlack"
                  >
                    Filter by status
                  </Text>
                  {hasActiveFilters ? (
                    <button
                      type="button"
                      onClick={clearStatuses}
                      className="text-xs font-medium text-purple-600 hover:text-purple-700"
                    >
                      Clear
                    </button>
                  ) : null}
                </div>
                <ul className="flex flex-col gap-1">
                  {FILTER_OPTIONS.map((option) => {
                    const checked = statuses.includes(option.value);
                    return (
                      <li key={option.value}>
                        <label
                          className={cn(
                            "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm text-zinc-700 transition hover:bg-zinc-50",
                            checked && "text-textBlack",
                          )}
                        >
                          <Checkbox
                            checked={checked}
                            onCheckedChange={() => toggleStatus(option.value)}
                            aria-label={option.label}
                          />
                          <span>{option.label}</span>
                        </label>
                      </li>
                    );
                  })}
                </ul>
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex-grow overflow-hidden pb-3">
            <h3 className="sr-only">List of agents</h3>
            <div
              className={cn(
                scrollbarStyles,
                "max-h-[44vh] min-h-[280px] overflow-y-auto pr-2",
              )}
              role="region"
              aria-labelledby="agentListHeading"
              aria-busy={isFetching}
            >
              <div id="agentListHeading" className="sr-only">
                Scrollable list of agents
              </div>
              {isLoading ? (
                <div className="grid grid-cols-1 gap-2 p-1 sm:grid-cols-2">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 rounded-[12px] border border-zinc-200 bg-white p-3"
                    >
                      <div className="flex flex-1 flex-col gap-2">
                        <Skeleton className="h-4 w-1/3" />
                        <Skeleton className="h-3 w-2/3" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : hasNoResults ? (
                <div className="flex min-h-[240px] flex-col items-center justify-center gap-2 rounded-[12px] border border-dashed border-zinc-200 px-6 py-8 text-center">
                  <Text variant="body-medium" className="text-textBlack">
                    No agents match the selected filters
                  </Text>
                  <Text variant="small" className="text-zinc-500">
                    Try a different combination or clear all filters.
                  </Text>
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={clearStatuses}
                  >
                    Clear filters
                  </Button>
                </div>
              ) : (
                <div
                  className={cn(
                    "grid grid-cols-1 gap-2 p-1 transition-opacity sm:grid-cols-2",
                    isFetching && "opacity-60",
                  )}
                >
                  {myAgents.map((agent) => {
                    const isSelected = selectedAgentId === agent.id;
                    return (
                      <button
                        type="button"
                        key={agent.id}
                        data-testid="agent-card"
                        className={cn(
                          "group flex w-full cursor-pointer select-none items-center gap-3 rounded-[12px] border bg-white p-3 text-left transition-[border-color,box-shadow] duration-150 hover:border-purple-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-offset-2",
                          isSelected
                            ? "border-purple-500 bg-purple-50/40 shadow-[0_0_0_3px_rgba(119,51,245,0.12)]"
                            : "border-zinc-200",
                        )}
                        onClick={() =>
                          handleAgentClick(agent.name, agent.id, agent.version)
                        }
                        aria-pressed={isSelected}
                      >
                        <div className="flex min-w-0 flex-1 flex-col gap-1">
                          <div className="flex min-w-0 items-center gap-2">
                            <Text
                              variant="body-medium"
                              as="span"
                              className="truncate text-textBlack"
                            >
                              {agent.name}
                            </Text>
                            <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] font-medium text-zinc-600">
                              v{agent.version}
                            </span>
                          </div>
                          <Text
                            variant="small"
                            as="span"
                            className="truncate text-zinc-500"
                          >
                            {agent.description
                              ? agent.description
                              : `Edited ${agent.lastEdited}`}
                          </Text>
                        </div>
                        {isSelected ? (
                          <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-purple-500 text-white">
                            <CheckCircleIcon size={14} weight="fill" />
                          </span>
                        ) : (
                          <span
                            aria-hidden
                            className="size-5 shrink-0 rounded-full border border-zinc-300"
                          />
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {totalPages > 1 ? (
            <PaginationBar
              page={page}
              totalPages={totalPages}
              totalItems={totalItems}
              pageSize={pageSize}
              onChange={goToPage}
            />
          ) : null}

          <StepFooter
            secondary={
              <Button
                variant="secondary"
                size="small"
                onClick={onCancel}
                className="w-full sm:w-auto"
              >
                Cancel
              </Button>
            }
            primary={
              <Button
                size="small"
                onClick={handleNext}
                disabled={isNextDisabled}
                className="w-full sm:w-auto"
              >
                Continue
              </Button>
            }
          />
        </>
      )}
    </div>
  );
}

interface PaginationBarProps {
  page: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onChange: (page: number) => void;
}

function PaginationBar({
  page,
  totalPages,
  totalItems,
  pageSize,
  onChange,
}: PaginationBarProps) {
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, totalItems);
  const pages = buildPageRange(page, totalPages);

  return (
    <div className="flex flex-col items-center justify-between gap-2 pb-3 pt-1 text-zinc-500 sm:flex-row">
      <Text variant="small" className="text-zinc-500">
        {start}–{end} of {totalItems}
      </Text>
      <div className="flex items-center gap-1">
        <button
          type="button"
          aria-label="Previous page"
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          className="flex size-8 items-center justify-center rounded-full border border-zinc-200 text-zinc-600 transition hover:border-zinc-300 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CaretLeftIcon size={14} weight="bold" />
        </button>
        {pages.map((entry, idx) =>
          entry === "…" ? (
            <span
              key={`gap-${idx}`}
              className="px-1 text-xs text-zinc-400"
              aria-hidden
            >
              …
            </span>
          ) : (
            <button
              key={entry}
              type="button"
              onClick={() => onChange(entry)}
              aria-current={entry === page ? "page" : undefined}
              className={cn(
                "flex size-8 items-center justify-center rounded-full border text-xs font-medium transition",
                entry === page
                  ? "border-purple-500 bg-purple-500 text-white"
                  : "border-zinc-200 text-zinc-600 hover:border-zinc-300",
              )}
            >
              {entry}
            </button>
          ),
        )}
        <button
          type="button"
          aria-label="Next page"
          disabled={page >= totalPages}
          onClick={() => onChange(page + 1)}
          className="flex size-8 items-center justify-center rounded-full border border-zinc-200 text-zinc-600 transition hover:border-zinc-300 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CaretRightIcon size={14} weight="bold" />
        </button>
      </div>
    </div>
  );
}

function buildPageRange(current: number, total: number): (number | "…")[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const pages: (number | "…")[] = [1];
  const left = Math.max(2, current - 1);
  const right = Math.min(total - 1, current + 1);
  if (left > 2) pages.push("…");
  for (let i = left; i <= right; i += 1) pages.push(i);
  if (right < total - 1) pages.push("…");
  pages.push(total);
  return pages;
}
