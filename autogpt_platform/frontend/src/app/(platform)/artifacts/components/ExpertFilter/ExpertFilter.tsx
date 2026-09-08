"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { LayoutGroup, motion, type Transition } from "framer-motion";

export interface ExpertFilterOption {
  id: string;
  name: string;
  avatarUrl: string | null;
}

interface Props {
  experts: ExpertFilterOption[];
  value: string | null;
  onChange: (expertId: string | null) => void;
}

const snappySpring: Transition = {
  type: "spring",
  stiffness: 350,
  damping: 30,
  mass: 1,
};

export function ExpertFilter({ experts, value, onChange }: Props) {
  // Keep the control while a selection exists so a filter on an expert who
  // has since been fired can still be cleared.
  if (experts.length === 0 && value === null) return null;

  return (
    <LayoutGroup id="artifacts-expert-filter">
      <div
        role="tablist"
        aria-label="Filter by expert"
        className="flex flex-wrap items-center gap-1"
        data-testid="artifacts-expert-filter"
      >
        <Text variant="small" as="span" className="px-2 text-zinc-500">
          From
        </Text>
        <ExpertTab
          label="Everyone"
          active={value === null}
          onClick={() => onChange(null)}
          testId="artifacts-expert-filter-everyone"
        />
        {experts.map((expert) => (
          <ExpertTab
            key={expert.id}
            label={expert.name}
            avatarUrl={expert.avatarUrl}
            active={value === expert.id}
            onClick={() => onChange(expert.id)}
            testId={`artifacts-expert-filter-${expert.id}`}
          />
        ))}
      </div>
    </LayoutGroup>
  );
}

interface ExpertTabProps {
  label: string;
  active: boolean;
  onClick: () => void;
  testId: string;
  /** Present for expert tabs; "Everyone" has no avatar. */
  avatarUrl?: string | null;
}

function ExpertTab({
  label,
  active,
  onClick,
  testId,
  avatarUrl,
}: ExpertTabProps) {
  const hasAvatar = avatarUrl !== undefined;
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "relative flex max-w-48 items-center gap-2 rounded-full py-1.5 text-sm font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400",
        hasAvatar ? "pl-1.5 pr-3.5" : "px-4",
        active ? "text-zinc-900" : "text-zinc-500 hover:text-zinc-900",
      )}
      data-testid={testId}
    >
      {active ? (
        <motion.span
          layoutId="artifacts-expert-active"
          className="absolute inset-0 rounded-full bg-zinc-100"
          transition={snappySpring}
        />
      ) : null}
      {hasAvatar ? (
        // Decorative: the label carries the name for assistive tech.
        <span aria-hidden className="relative z-10 shrink-0">
          <ExpertAvatar name={label} avatarUrl={avatarUrl} size={22} />
        </span>
      ) : null}
      <span className="relative z-10 truncate">{label}</span>
    </button>
  );
}
