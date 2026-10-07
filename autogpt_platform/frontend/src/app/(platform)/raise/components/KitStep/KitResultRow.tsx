"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { FlowIcon, SparklesIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { MAX_ATTACHMENTS, type SearchHit } from "./helpers";

interface Props {
  hit: SearchHit;
  index: number;
  selected: boolean;
  atCap: boolean;
  isPending: boolean;
  onAdd: () => void;
}

export function KitResultRow({
  hit,
  index,
  selected,
  atCap,
  isPending,
  onAdd,
}: Props) {
  return (
    <div
      role="listitem"
      // Staggered so the list resolves as a cascade instead of a hard swap
      // from the skeleton; capped at 3 rows so the last one is never late.
      style={{ animationDelay: `${index * 60}ms` }}
      className={cn(
        "flex items-center gap-3 border-b border-border px-3.5 py-3 transition-colors duration-200 last:border-b-0 hover:bg-zinc-50",
        "animate-in duration-300 fill-mode-both fade-in slide-in-from-bottom-1 motion-reduce:animate-none",
      )}
    >
      <span
        aria-hidden
        className="grid size-9 shrink-0 place-items-center rounded-xl border border-border bg-zinc-50 text-muted-foreground"
      >
        <Icon icon={hit.kind === "skill" ? SparklesIcon : FlowIcon} size={16} />
      </span>

      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          unmask={false}
          className="truncate text-foreground"
        >
          {hit.name}
        </Text>
        <Text variant="small" tone="muted" unmask={false} className="truncate">
          {hit.subtitle}
        </Text>
      </div>

      <Button
        type="button"
        variant={selected ? "ghost" : "secondary"}
        size="md"
        disabled={selected || atCap}
        loading={isPending}
        onClick={onAdd}
        className={cn(
          "shrink-0 rounded-xl transition-all duration-200",
          // The ghost variant greys disabled text down to zinc-200, which
          // reads as broken rather than settled — keep "Added" full strength.
          selected && "disabled:text-zinc-800",
        )}
        leftIcon={
          selected ? (
            // The tick scales in as the spinner it replaced fades out, so
            // adding reads as one motion.
            <Icon
              icon={Tick02Icon}
              size={14}
              aria-hidden
              className="animate-in duration-200 zoom-in-50 fade-in motion-reduce:animate-none"
            />
          ) : undefined
        }
      >
        {selected ? "Added" : atCap ? `${MAX_ATTACHMENTS} max` : "Add"}
      </Button>
    </div>
  );
}
