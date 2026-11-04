import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import type { TaskInboxStep } from "../../../data/types";
import { relativeTime, stepDotClass } from "../helpers";

interface Props {
  steps: TaskInboxStep[];
  now: number;
}

export function StepList({ steps, now }: Props) {
  if (!steps.length) return null;
  return (
    <ol aria-label="Latest steps" className="flex flex-col gap-1.5">
      {steps.map((step, index) => {
        const live = step.status === "running";
        return (
          <li
            key={`${step.time}-${index}`}
            className="flex min-w-0 items-center gap-2"
          >
            <span
              aria-hidden
              className="inline-flex size-3 shrink-0 items-center justify-center"
            >
              {live ? (
                <Icon icon={Loading03Icon} size={12} tone="info" spin />
              ) : (
                <span
                  className={cn("size-1.5 rounded-full", stepDotClass(step))}
                />
              )}
            </span>
            <span
              className={cn(
                "flex min-w-0 flex-1",
                live && "animate-pulse motion-reduce:animate-none",
              )}
            >
              <Text
                as="span"
                variant="inbox"
                tone={live ? "ink" : "text"}
                className="truncate"
              >
                {step.label}
              </Text>
            </span>
            <Text
              as="span"
              variant="small"
              tone="faint"
              tabular
              className="shrink-0 whitespace-nowrap"
            >
              {live ? "now" : relativeTime(step.time, now)}
            </Text>
          </li>
        );
      })}
    </ol>
  );
}
