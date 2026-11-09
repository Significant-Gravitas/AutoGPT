import type { WorkflowDiagramBlock } from "@/app/api/__generated__/models/workflowDiagramBlock";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { BlockCard } from "./BlockCard";

const STATUS_CLASSES: Record<string, string> = {
  ok: "border-border-base bg-surface-muted",
  idle: "border-border-base bg-surface-muted",
  warning: "border-warning-border bg-warning-surface",
  error: "border-error bg-error-surface",
};

// Colour alone doesn't reach everyone, so a step in trouble says so.
const STATUS_LABELS: Record<string, string> = {
  warning: "Needs attention",
  error: "Problem",
};

interface Props {
  block: WorkflowDiagramBlock;
}

export function WorkflowDiagramBlockView({ block }: Props) {
  return (
    <BlockCard title={block.title}>
      <ol className="flex flex-wrap items-stretch gap-y-2">
        {block.steps.map((step, index) => {
          const status = step.status ?? "idle";
          const statusLabel = STATUS_LABELS[status];
          return (
            <li key={index} className="flex min-w-0 flex-1 items-center">
              <div
                className={cn(
                  "h-full min-w-24 flex-1 rounded-lg border px-2.5 py-1.5",
                  STATUS_CLASSES[status],
                )}
              >
                <Text variant="small" tone="primary" className="break-words">
                  {step.label}
                </Text>
                {step.detail ? (
                  <Text
                    variant="small"
                    tone={status === "error" ? "error-strong" : "muted"}
                    className="break-words"
                  >
                    {step.detail}
                  </Text>
                ) : null}
                {statusLabel ? (
                  <span className="sr-only">{statusLabel}</span>
                ) : null}
              </div>
              {index < block.steps.length - 1 ? (
                <Icon
                  icon={ArrowRight01Icon}
                  size={14}
                  tone="muted"
                  className="mx-0.5 shrink-0"
                />
              ) : null}
            </li>
          );
        })}
      </ol>
    </BlockCard>
  );
}
