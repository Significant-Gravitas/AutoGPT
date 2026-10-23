import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import { motion, useReducedMotion } from "framer-motion";

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
                className="absolute bottom-0 left-[1.125rem] top-10 w-px -translate-x-1/2 bg-zinc-200"
              />
            )}
            <span
              aria-hidden
              className={cn(
                "relative flex size-9 items-center justify-center rounded-full",
                isFirst
                  ? "bg-[#B6A4C8]/[0.24] text-zinc-900"
                  : "border border-zinc-200 bg-white text-zinc-500",
              )}
            >
              <Icon icon={milestone.icon} size={16} />
            </span>
            <div className={cn("flex flex-col", !isLast && "pb-5")}>
              <Text variant="small-medium" as="span" tone="secondary">
                {milestone.date ? (
                  <time dateTime={milestone.date.toISOString().slice(0, 10)}>
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
