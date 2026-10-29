import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  label: string;
  icon: IconSvgElement;
  value: number | null;
  detail?: string;
  isLoading?: boolean;
}

/** A headline number: the figure carries the tile, so no chart. */
export function StatTile({ label, icon, value, detail, isLoading }: Props) {
  return (
    <div className="flex flex-col gap-1 rounded-2xl bg-surface p-4 smooth-shadow-ring-sm">
      <div className="flex items-center gap-1.5 text-foreground-tertiary">
        <Icon icon={icon} size={14} aria-hidden />
        <Text variant="small-medium" tone="muted">
          {label}
        </Text>
      </div>
      {isLoading ? (
        <Skeleton className="mt-1 h-8 w-16 rounded-md" />
      ) : (
        <p className="font-sans text-heading font-semibold tabular-nums leading-9 tracking-title text-foreground-strong">
          {value ?? "—"}
        </p>
      )}
      {detail ? (
        <Text variant="small" tone="muted">
          {detail}
        </Text>
      ) : null}
    </div>
  );
}
