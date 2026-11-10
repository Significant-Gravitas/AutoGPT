import type { TimelineBlock } from "@/app/api/__generated__/models/timelineBlock";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { BlockCard } from "./BlockCard";

const DOT_CLASSES: Record<string, string> = {
  ok: "bg-success",
  warning: "bg-warning",
  error: "bg-error",
  idle: "bg-border-strong",
};

// Colour alone doesn't reach everyone, so a step's state is also spoken.
const STATUS_LABELS: Record<string, string> = {
  ok: "Done",
  warning: "Needs attention",
  error: "Problem",
};

interface Props {
  block: TimelineBlock;
}

export function TimelineBlockView({ block }: Props) {
  const lastIndex = block.items.length - 1;

  return (
    <BlockCard title={block.title}>
      <ol>
        {block.items.map((item, index) => {
          const status = item.status ?? "idle";
          const statusLabel = STATUS_LABELS[status];
          return (
            <li key={index} className="relative flex gap-3.5 pb-5 last:pb-0">
              {index < lastIndex ? (
                <span
                  aria-hidden
                  className="absolute top-4 bottom-0 left-[0.3125rem] w-px bg-border-base"
                />
              ) : null}
              <span aria-hidden className="flex w-2.5 shrink-0 pt-1.5">
                <span
                  className={cn(
                    "size-2.5 rounded-full ring-4 ring-surface",
                    DOT_CLASSES[status],
                  )}
                />
              </span>
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <div className="flex items-start justify-between gap-3">
                  <Text
                    variant="body-medium"
                    tone="primary"
                    className="min-w-0 break-words"
                  >
                    {item.title}
                    {statusLabel ? (
                      <span className="sr-only">, {statusLabel}</span>
                    ) : null}
                  </Text>
                  {item.date ? (
                    <span className="shrink-0 rounded-full bg-surface-hover px-2 py-0.5">
                      <Text
                        as="span"
                        variant="small-medium"
                        tone="secondary"
                        tabular
                      >
                        {item.date}
                      </Text>
                    </span>
                  ) : null}
                </div>
                {item.detail ? (
                  <Text
                    variant="body-tight"
                    tone="secondary"
                    className="break-words"
                  >
                    {item.detail}
                  </Text>
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </BlockCard>
  );
}
