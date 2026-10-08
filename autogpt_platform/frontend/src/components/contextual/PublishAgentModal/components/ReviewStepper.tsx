import { motion } from "motion/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  Clock01Icon,
  Rocket01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

type StepState = "done" | "current" | "upcoming";

interface Step {
  title: string;
  description: string;
  note?: string;
  state: StepState;
  Icon: IconSvgElement;
}

const STEPS: Step[] = [
  {
    title: "Submitted for review",
    description: "Your listing is queued in the marketplace review pipeline.",
    state: "done",
    Icon: Tick02Icon,
  },
  {
    title: "In review",
    description:
      "Our team checks the details, media, and safety of your agent.",
    note: "Typically reviewed within 2–3 days.",
    state: "current",
    Icon: Clock01Icon,
  },
  {
    title: "Goes live",
    description:
      "You'll get an email once it's approved. Rejected listings come back with feedback.",
    state: "upcoming",
    Icon: Rocket01Icon,
  },
];

const NODE_CLASS: Record<StepState, string> = {
  done: "bg-green-500 text-white",
  current: "bg-yellow-50 text-yellow-700 ring-2 ring-yellow-300",
  upcoming: "bg-zinc-100 text-zinc-400",
};

interface Props {
  shouldReduceMotion: boolean;
}

export function ReviewStepper({ shouldReduceMotion }: Props) {
  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24, ease: "easeOut", delay: 0.24 }}
      className="mt-6 flex w-full max-w-md flex-col gap-3 px-2"
    >
      <Text variant="body-medium" as="span">
        What happens next
      </Text>
      <ol className="flex flex-col">
        {STEPS.map((step, index) => {
          const isLast = index === STEPS.length - 1;
          return (
            <li key={step.title} className="flex gap-3">
              <div className="flex flex-col items-center">
                <span
                  className={cn(
                    "relative mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full",
                    NODE_CLASS[step.state],
                  )}
                >
                  <Icon icon={step.Icon} size={14} />
                </span>
                {!isLast ? (
                  <span
                    className={cn(
                      "mt-1 min-h-4.5 w-px flex-1",
                      step.state === "done" ? "bg-green-300" : "bg-zinc-200",
                    )}
                  />
                ) : null}
              </div>
              <div className={cn("flex min-w-0 flex-col", !isLast && "pb-4")}>
                <Text
                  variant="small-medium"
                  as="span"
                  className={cn(step.state === "upcoming" && "text-zinc-400")}
                >
                  {step.title}
                </Text>
                <Text
                  variant="small"
                  tone="muted"
                  className={cn(step.state === "upcoming" && "text-zinc-400")}
                >
                  {step.description}
                </Text>
                {step.note ? (
                  <Text variant="small" className="mt-1 text-yellow-700">
                    {step.note}
                  </Text>
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </motion.div>
  );
}
