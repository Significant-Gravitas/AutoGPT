import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import { motion, useReducedMotion } from "framer-motion";
import { OTTO_TINT } from "./ottoTint";

export interface Milestone {
  key: string;
  icon: IconSvgElement;
  when: string;
  date?: Date;
  title: string;
  detail: string;
}

const EASE = [0.22, 1, 0.36, 1] as const;

interface Props {
  label: string;
  milestones: Milestone[];
}

export function Timeline({ label, milestones }: Props) {
  const reduceMotion = useReducedMotion();
  return (
    <ol aria-label={label} className="flex flex-col">
      {milestones.map((milestone, index) => {
        const isFirst = index === 0;
        const isLast = index === milestones.length - 1;
        return (
          <motion.li
            key={milestone.key}
            initial={reduceMotion ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              delay: 0.15 + index * 0.12,
              duration: 0.45,
              ease: EASE,
            }}
            className="relative grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-4"
          >
            {isLast ? null : (
              <span
                aria-hidden
                className="absolute bottom-0 left-[1.125rem] top-10 w-px -translate-x-1/2 bg-border-base"
              />
            )}
            <span
              aria-hidden
              style={isFirst ? { backgroundColor: OTTO_TINT } : undefined}
              className={cn(
                "relative flex size-9 items-center justify-center rounded-full",
                isFirst
                  ? "text-foreground-strong"
                  : "border border-border-base bg-surface text-foreground-tertiary",
              )}
            >
              <Icon icon={milestone.icon} size={16} />
            </span>
            <div className={cn("flex flex-col", !isLast && "pb-5")}>
              <Text variant="small-medium" as="span" tone="secondary">
                {milestone.date ? (
                  <time dateTime={toLocalISODate(milestone.date)}>
                    {milestone.when}
                  </time>
                ) : (
                  milestone.when
                )}
              </Text>
              <Text variant="large-medium" as="span" tone="primary">
                {milestone.title}
              </Text>
              <Text variant="body" tone="secondary" className="mt-0.5">
                {milestone.detail}
              </Text>
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}

// The visible date is local, so the machine-readable one must be too.
function toLocalISODate(date: Date) {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}
