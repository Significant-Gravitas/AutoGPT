"use client";

import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useFirstJobStep } from "./useFirstJobStep";

interface Props {
  onPick: (job: { id: string; name: string }) => void;
  onSkip: () => void;
}

export function FirstJobStep({ onPick, onSkip }: Props) {
  const {
    suggestions,
    isLoading,
    hasError,
    selected,
    select,
    isResolving,
    canConfirm,
    confirm,
    retry,
  } = useFirstJobStep({ onPick });

  return (
    <div className="flex flex-col gap-3">
      {hasError ? (
        <ErrorCard
          context="starter jobs"
          hint="We couldn't load the available jobs. You can retry or skip for now."
          onRetry={retry}
        />
      ) : isLoading ? (
        <div className="flex flex-col gap-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-20 w-full rounded-2xl" />
          ))}
        </div>
      ) : suggestions.length > 0 ? (
        suggestions.map((agent) => (
          <JobCard
            key={agent.slug}
            agent={agent}
            isSelected={selected?.slug === agent.slug}
            onSelect={() => select(agent)}
          />
        ))
      ) : (
        <p className="rounded-2xl border border-zinc-200 bg-white p-5 text-sm text-zinc-500">
          No starter jobs are available right now. You can skip for now.
        </p>
      )}

      <footer className="flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onSkip}>
          Skip for now
        </Button>
        <Button
          variant="primary"
          onClick={confirm}
          disabled={!canConfirm}
          loading={isResolving}
          className="rounded-full"
        >
          Give me this job
        </Button>
      </footer>
    </div>
  );
}

interface JobCardProps {
  agent: StoreAgent;
  isSelected: boolean;
  onSelect: () => void;
}

function JobCard({ agent, isSelected, onSelect }: JobCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={isSelected}
      className={cn(
        "w-full rounded-2xl border p-4 text-left transition-colors",
        isSelected
          ? "border-purple-300 bg-purple-50/40 ring-2 ring-purple-200"
          : "border-zinc-200 bg-white hover:border-zinc-300",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="text-[15px] font-medium text-zinc-900">
          {agent.agent_name}
        </span>
        {isSelected ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={18}
            className="shrink-0 text-purple-600"
          />
        ) : null}
      </div>
      {agent.sub_heading ? (
        <p className="mt-1 line-clamp-2 text-sm text-zinc-500">
          {agent.sub_heading}
        </p>
      ) : null}
    </button>
  );
}
