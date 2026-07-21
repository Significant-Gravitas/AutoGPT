"use client";

import { cn } from "@/lib/utils";
import {
  ListBulletsIcon,
  SparkleIcon,
  UploadSimpleIcon,
  type Icon,
} from "@phosphor-icons/react";
import { LayoutGroup, motion, type Transition } from "framer-motion";
import type { OriginFilter as OriginFilterValue } from "../../useArtifactsPage";

interface Props {
  value: OriginFilterValue;
  onChange: (value: OriginFilterValue) => void;
}

interface Option {
  value: OriginFilterValue;
  label: string;
  Icon: Icon;
}

const OPTIONS: Option[] = [
  { value: "all", label: "All", Icon: ListBulletsIcon },
  { value: "uploaded", label: "Uploaded", Icon: UploadSimpleIcon },
  { value: "generated", label: "Generated", Icon: SparkleIcon },
];

const snappySpring: Transition = {
  type: "spring",
  stiffness: 350,
  damping: 30,
  mass: 1,
};

export function OriginFilter({ value, onChange }: Props) {
  return (
    <LayoutGroup id="artifacts-origin-filter">
      <div
        role="tablist"
        aria-label="Filter by source"
        className="inline-flex items-center gap-1 rounded-full border border-zinc-200 bg-zinc-50 p-1"
        data-testid="artifacts-origin-filter"
      >
        {OPTIONS.map((opt) => (
          <OriginTab
            key={opt.value}
            option={opt}
            active={value === opt.value}
            onClick={() => onChange(opt.value)}
          />
        ))}
      </div>
    </LayoutGroup>
  );
}

interface OriginTabProps {
  option: Option;
  active: boolean;
  onClick: () => void;
}

function OriginTab({ option, active, onClick }: OriginTabProps) {
  const { label, Icon } = option;
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "relative flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium outline-none transition-colors",
        active ? "text-white" : "text-zinc-500 hover:text-zinc-800",
      )}
      data-testid={`artifacts-origin-filter-${option.value}`}
    >
      {active ? (
        <motion.span
          layoutId="artifacts-origin-active"
          className="absolute inset-0 rounded-full bg-zinc-900 shadow-sm"
          transition={snappySpring}
        />
      ) : null}
      <span className="relative z-10 flex items-center gap-1.5">
        <Icon
          size={14}
          className={cn(
            "transition-transform duration-300",
            active && "scale-110",
          )}
        />
        {label}
      </span>
    </button>
  );
}
