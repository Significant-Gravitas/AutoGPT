"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  CaretLeftIcon,
  CaretRightIcon,
  CheckCircleIcon,
  PlusIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";

import { Text } from "../../../../atoms/Text/Text";
import { Button } from "../../../../atoms/Button/Button";
import { Select } from "../../../../atoms/Select/Select";
import { StepHeader } from "../StepHeader";
import { StepFooter } from "../StepFooter";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { useAgentSelectStep } from "./useAgentSelectStep";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { MyAgentsSortBy } from "@/app/api/__generated__/models/myAgentsSortBy";

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
    pageDirection,
    handleAgentClick,
    handleNext,
    handleSortChange,
    goToPage,
    isNextDisabled,
  } = useAgentSelectStep({ onSelect, onNext });

  const shouldReduceMotion = useReducedMotion();

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

  const showEmpty = !isLoading && totalItems === 0 && myAgents.length === 0;

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
          <div className="mt-2 flex items-center justify-start pb-0">
            <div className="w-full sm:w-[220px]">
              <Select
                id="agent-sort"
                label="Sort agents"
                hideLabel
                size="small"
                value={sortBy}
                onValueChange={handleSortChange}
                options={SORT_OPTIONS}
                wrapperClassName="mb-2"
              />
            </div>
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
              ) : (
                <AnimatePresence
                  mode="wait"
                  initial={false}
                  custom={pageDirection}
                >
                  <motion.div
                    key={page}
                    custom={pageDirection}
                    variants={{
                      enter: (dir: number) => ({
                        opacity: 0,
                        x: shouldReduceMotion ? 0 : dir * 16,
                      }),
                      center: {
                        opacity: 1,
                        x: 0,
                        transition: {
                          duration: shouldReduceMotion ? 0 : 0.2,
                          ease: [0.16, 1, 0.3, 1],
                        },
                      },
                      exit: (dir: number) => ({
                        opacity: 0,
                        x: shouldReduceMotion ? 0 : dir * -16,
                        transition: {
                          duration: shouldReduceMotion ? 0 : 0.12,
                          ease: [0.4, 0, 1, 1],
                        },
                      }),
                    }}
                    initial="enter"
                    animate="center"
                    exit="exit"
                    className={cn(
                      "grid grid-cols-1 gap-2 p-1 sm:grid-cols-2",
                      isFetching && "opacity-90",
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
                            handleAgentClick(
                              agent.name,
                              agent.id,
                              agent.version,
                            )
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
                  </motion.div>
                </AnimatePresence>
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
    <div className="flex flex-col items-center justify-between gap-3 pb-3 pt-1 text-zinc-500 sm:flex-row">
      <Text variant="small" className="text-zinc-500">
        {start}–{end} of {totalItems}
      </Text>
      <div className="flex items-center gap-2">
        <button
          type="button"
          aria-label="Previous page"
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          className="flex size-9 items-center justify-center rounded-full border border-zinc-200 text-zinc-500 transition hover:border-zinc-300 hover:text-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CaretLeftIcon size={14} weight="bold" />
        </button>
        {pages.map((entry, idx) =>
          entry === "…" ? (
            <span
              key={`gap-${idx}`}
              className="px-1 text-sm text-zinc-400"
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
                "relative flex size-9 items-center justify-center rounded-full text-sm font-medium transition",
                entry === page
                  ? "text-white"
                  : "border border-zinc-200 text-zinc-700 hover:border-zinc-300",
              )}
            >
              {entry === page ? (
                <motion.span
                  layoutId="pagination-active-pill"
                  aria-hidden
                  className="absolute inset-0 rounded-full bg-purple-500 shadow-[0_4px_10px_-4px_rgba(119,51,245,0.55)]"
                  transition={{ type: "spring", stiffness: 420, damping: 32 }}
                />
              ) : null}
              <span className="relative">{entry}</span>
            </button>
          ),
        )}
        <button
          type="button"
          aria-label="Next page"
          disabled={page >= totalPages}
          onClick={() => onChange(page + 1)}
          className="flex size-9 items-center justify-center rounded-full border border-zinc-200 text-zinc-500 transition hover:border-zinc-300 hover:text-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
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
