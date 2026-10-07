"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { LayoutGroup, motion, type Transition } from "framer-motion";
import type { OriginFilter as OriginFilterValue } from "../../useArtifactsPage";

interface Props {
  value: OriginFilterValue;
  onChange: (value: OriginFilterValue) => void;
}

interface Option {
  value: OriginFilterValue;
  label: string;
}

const OPTIONS: Option[] = [
  { value: "all", label: "All" },
  { value: "uploaded", label: "Uploaded" },
  { value: "generated", label: "Generated" },
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
      <div className="flex flex-col gap-1">
        <Text variant="small" as="span" className="pl-4 text-muted-foreground">
          Type
        </Text>
        <div
          role="tablist"
          aria-label="Filter by type"
          className="flex items-center gap-1"
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
  return (
    <Button
      type="button"
      variant="ghost"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "relative h-auto min-w-0 rounded-full border-0 px-4 py-2 text-sm font-medium outline-hidden hover:bg-transparent focus-visible:ring-2 focus-visible:ring-zinc-400",
        active ? "text-zinc-900" : "text-muted-foreground hover:text-zinc-900",
      )}
      data-testid={`artifacts-origin-filter-${option.value}`}
    >
      {active ? (
        <motion.span
          layoutId="artifacts-origin-active"
          className="absolute inset-0 rounded-full bg-zinc-100"
          transition={snappySpring}
        />
      ) : null}
      <span className="relative z-10">{option.label}</span>
    </Button>
  );
}
