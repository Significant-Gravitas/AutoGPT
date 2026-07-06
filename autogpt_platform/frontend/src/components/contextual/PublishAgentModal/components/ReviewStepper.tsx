import { motion } from "framer-motion";
import {
  CheckIcon,
  ClockIcon,
  RocketLaunchIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type StepState = "done" | "current" | "upcoming";

interface Step {
  title: string;
  description: string;
  note?: string;
  state: StepState;
  Icon: PhosphorIcon;
}

const STEPS: Step[] = [
  {
    title: "Submitted for review",
    description: "Your listing is queued in the marketplace review pipeline.",
    state: "done",
    Icon: CheckIcon,
  },
  {
    title: "In review",
    description:
      "Our team checks the details, media, and safety of your agent.",
    note: "Typically reviewed within 2–3 days.",
    state: "current",
    Icon: ClockIcon,
  },
  {
    title: "Goes live",
    description:
      "You'll get an email once it's approved. Rejected listings come back with feedback.",
    state: "upcoming",
    Icon: RocketLaunchIcon,
  },
];

const NODE_CLASS: Record<StepState, string> = {
  done: "bg-emerald-500 text-white",
  current: "bg-amber-50 text-amber-700 ring-2 ring-amber-300",
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
      <Text variant="body-medium" as="span" className="text-textBlack">
        What happens next
      </Text>
      <ol className="flex flex-col">
        {STEPS.map((step, index) => {
          const isLast = index === STEPS.length - 1;
          const StepIcon = step.Icon;
          return (
            <li key={step.title} className="flex gap-3">
              <div className="flex flex-col items-center">
                <span
                  className={cn(
                    "relative mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full",
                    NODE_CLASS[step.state],
                  )}
                >
                  <StepIcon size={14} weight="bold" />
                </span>
                {!isLast ? (
                  <span
                    className={cn(
                      "mt-1 min-h-[18px] w-px flex-1",
                      step.state === "done" ? "bg-emerald-300" : "bg-zinc-200",
                    )}
                  />
                ) : null}
              </div>
              <div className={cn("flex min-w-0 flex-col", !isLast && "pb-4")}>
                <Text
                  variant="small-medium"
                  as="span"
                  className={
                    step.state === "upcoming"
                      ? "text-zinc-400"
                      : "text-textBlack"
                  }
                >
                  {step.title}
                </Text>
                <Text
                  variant="small"
                  className={
                    step.state === "upcoming"
                      ? "text-zinc-400"
                      : "text-zinc-500"
                  }
                >
                  {step.description}
                </Text>
                {step.note ? (
                  <Text variant="small" className="mt-1 text-amber-700">
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
