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
 *  marketplace card's footer instead of a stats grid. Kept to a single
 *  line: 12px text, and the words give way before the counts do. */
export function CardStats({ className, children }: Props) {
  return (
    <dl
      className={cn(
        "flex flex-nowrap items-center gap-x-3 overflow-hidden",
        className,
      )}
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
        "flex min-w-0 items-center gap-1",
        isZero ? "text-zinc-400" : "text-zinc-500",
      )}
    >
      <Icon icon={icon} size={13} className="shrink-0" aria-hidden="true" />
      <Text
        variant="small-medium"
        as="dd"
        unmask={false}
        className={cn(
          "shrink-0 tabular-nums",
          isZero ? "!text-zinc-400" : "!text-zinc-800",
        )}
      >
        {count}
      </Text>
      <Text variant="small" as="dt" className="min-w-0 truncate !text-inherit">
        <span className="sr-only">{label}</span>
        <span aria-hidden="true">{word}</span>
      </Text>
    </div>
  );
}
