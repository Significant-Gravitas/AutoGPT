import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface Props {
  className?: string;
  children: ReactNode;
}

/** A card's totals as one tinted pill, each stat a value over its label. */
export function CardStats({ className, children }: Props) {
  return (
    <dl
      className={cn(
        "flex items-stretch rounded-lg bg-zinc-50 px-2 py-2 ring-1 ring-inset ring-zinc-100",
        className,
      )}
    >
      {children}
    </dl>
  );
}

interface StatProps {
  label: string;
  children: ReactNode;
}

export function CardStat({ label, children }: StatProps) {
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center gap-0.5 text-center">
      <Text
        variant="small"
        as="dt"
        tone="muted"
        className="order-2 w-full truncate"
      >
        {label}
      </Text>
      <Text
        variant="large-semibold"
        as="dd"
        tone="primary"
        unmask={false}
        className="order-1 tabular-nums"
      >
        {children}
      </Text>
    </div>
  );
}
