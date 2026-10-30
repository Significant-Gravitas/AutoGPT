import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  AUTOPILOT_HEX,
  expertPastel,
} from "@/components/molecules/ExpertAvatar/colors";
import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import { type Milestone, toLocalISODate } from "./helpers";

const EASE = [0.22, 1, 0.36, 1] as const;
// Otto's lavender at the design system's 24% tint, behind the first row.
const OTTO_TINT = expertPastel(AUTOPILOT_HEX);

interface Props {
  label: string;
  milestones: Milestone[];
}

export function TrialTimeline({ label, milestones }: Props) {
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
            transition={{ delay: 0.1 + index * 0.1, duration: 0.4, ease: EASE }}
            className="relative grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-3.5"
          >
            {isLast ? null : (
              <span
                aria-hidden
                className="absolute top-10 bottom-0 left-[1.125rem] w-px -translate-x-1/2 bg-border-base"
              />
            )}
            <span
              aria-hidden
              // oxlint-disable-next-line design-system/no-inline-styles -- Otto's tint is mixed from the expert palette (ExpertAvatar/colors); no role has it
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
            <div
              className={cn(
                "flex flex-col",
                !isLast && "pb-4 [@media(max-height:780px)]:pb-3",
              )}
            >
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
              <Text
                variant="body"
                tone="secondary"
                className="mt-0.5 text-pretty"
              >
                {milestone.detail}
              </Text>
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}
