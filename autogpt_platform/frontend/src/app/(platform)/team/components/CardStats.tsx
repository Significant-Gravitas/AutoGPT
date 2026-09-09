import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import { ReactNode } from "react";

interface Props {
  className?: string;
  children: ReactNode;
}

/** A card's totals as one meta line: icon, count, word. Reads like the
 *  marketplace card's footer instead of a stats grid. */
export function CardStats({ className, children }: Props) {
  return (
    <dl
      className={cn("flex flex-wrap items-center gap-x-3 gap-y-1", className)}
    >
      {children}
    </dl>
  );
}

interface StatProps {
  icon: IconSvgElement;
  /** Plural noun, also the accessible label: "Schedules". */
  label: string;
  singular: string;
  count: number;
}

export function CardStat({ icon, label, singular, count }: StatProps) {
  const isZero = count === 0;
  const word = count === 1 ? singular : label.toLowerCase();

  return (
    <div
      className={cn(
        "flex items-center gap-1",
        isZero ? "text-zinc-400" : "text-zinc-500",
      )}
    >
      <Icon icon={icon} size={14} className="shrink-0" aria-hidden="true" />
      <Text
        variant="body-medium"
        as="dd"
        unmask={false}
        className={cn(
          "tabular-nums",
          isZero ? "!text-zinc-400" : "!text-zinc-800",
        )}
      >
        {count}
      </Text>
      <Text variant="body" as="dt" className="!text-inherit">
        <span className="sr-only">{label}</span>
        <span aria-hidden="true">{word}</span>
      </Text>
    </div>
  );
}
