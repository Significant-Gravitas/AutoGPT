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

interface Props {
  block: TimelineBlock;
}

export function TimelineBlockView({ block }: Props) {
  return (
    <BlockCard title={block.title}>
      <ol className="flex flex-col gap-3">
        {block.items.map((item, index) => (
          <li key={index} className="flex min-w-0 gap-3">
            <span
              aria-hidden
              className={cn(
                "mt-1.5 size-2 shrink-0 rounded-full",
                DOT_CLASSES[item.status ?? "idle"],
              )}
            />
            <div className="min-w-0">
              {item.date ? (
                <Text variant="small" tone="muted">
                  {item.date}
                </Text>
              ) : null}
              <Text
                variant="body-medium"
                tone="primary"
                className="break-words"
              >
                {item.title}
              </Text>
              {item.detail ? (
                <Text variant="small" tone="secondary" className="break-words">
                  {item.detail}
                </Text>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </BlockCard>
  );
}
