import type { WorkflowDiagramBlock } from "@/app/api/__generated__/models/workflowDiagramBlock";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { BlockCard } from "./BlockCard";

const STATUS_CLASSES: Record<string, string> = {
  ok: "from-surface to-surface-muted ring-1 ring-border-base",
  idle: "from-surface to-surface-muted ring-1 ring-border-base",
  warning: "from-warning-surface to-surface ring-1 ring-warning-border",
  error: "from-error-surface to-surface ring-1 ring-error/40",
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
      {/* A column on phones, where four steps side by side would not fit.
          Wider, each step keeps room for its card and arrow, and the row
          wraps once they no longer fit. */}
      <ol className="flex flex-col sm:flex-row sm:flex-wrap sm:items-stretch sm:gap-y-2">
        {block.steps.map((step, index) => {
          const status = step.status ?? "idle";
          const statusLabel = STATUS_LABELS[status];
          return (
            <li
              key={index}
              className="flex min-w-0 flex-col items-stretch sm:min-w-36 sm:flex-1 sm:flex-row sm:items-center"
            >
              <div
                className={cn(
                  "min-w-0 flex-1 rounded-xl bg-gradient-to-b px-3 py-2.5 sm:h-full",
                  STATUS_CLASSES[status],
                )}
              >
                <Text
                  variant="body-medium"
                  tone="primary"
                  className="break-words"
                >
                  {step.label}
                </Text>
                {step.detail ? (
                  <Text
                    variant="small-medium"
                    tone={status === "error" ? "error-strong" : "muted"}
                    tabular
                    className="mt-0.5 break-words"
                  >
                    {step.detail}
                  </Text>
                ) : null}
                {statusLabel ? (
                  <span className="sr-only">{statusLabel}</span>
                ) : null}
              </div>
              {index < block.steps.length - 1 ? (
                <>
                  <Icon
                    icon={ArrowDown01Icon}
                    size={14}
                    tone="subtle"
                    className="my-1 self-center sm:hidden"
                  />
                  <Icon
                    icon={ArrowRight01Icon}
                    size={14}
                    tone="subtle"
                    className="mx-1 hidden shrink-0 sm:block"
                  />
                </>
              ) : null}
            </li>
          );
        })}
      </ol>
    </BlockCard>
  );
}
