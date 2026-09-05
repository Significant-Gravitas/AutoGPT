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
      <dt className="order-2 w-full">
        <Text variant="small" className="truncate text-zinc-500">
          {label}
        </Text>
      </dt>
      <dd className="order-1">
        <Text
          variant="large-semibold"
          unmask={false}
          className="tabular-nums text-zinc-900"
        >
          {children}
        </Text>
      </dd>
    </div>
  );
}
