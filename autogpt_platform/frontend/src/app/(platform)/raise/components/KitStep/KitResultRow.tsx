"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
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
        "duration-300 animate-in fade-in slide-in-from-bottom-1 fill-mode-both motion-reduce:animate-none",
      )}
    >
      <span
        aria-hidden
        className="grid size-9 shrink-0 place-items-center rounded-xl border border-border bg-zinc-50 text-muted-foreground"
      >
        <Icon icon={hit.kind === "skill" ? SparklesIcon : FlowIcon} size={16} />
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">
          {hit.name}
        </p>
        <p className="truncate text-xs text-muted-foreground">{hit.subtitle}</p>
      </div>

      <Button
        type="button"
        variant={selected ? "ghost" : "secondary"}
        size="small"
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
              className="duration-200 animate-in fade-in zoom-in-50 motion-reduce:animate-none"
            />
          ) : undefined
        }
      >
        {selected ? "Added" : atCap ? `${MAX_ATTACHMENTS} max` : "Add"}
      </Button>
    </div>
  );
}
