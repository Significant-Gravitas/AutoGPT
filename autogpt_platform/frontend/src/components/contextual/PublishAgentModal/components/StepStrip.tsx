import { Fragment } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { CheckIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type Props = {
  currentStep: "select" | "info" | "review";
};

const steps = [
  { key: "select", label: "Agent" },
  { key: "info", label: "Listing" },
  { key: "review", label: "Review" },
] as const;

function getStepIndex(currentStep: Props["currentStep"]) {
  return steps.findIndex((step) => step.key === currentStep);
}

export function StepStrip({ currentStep }: Props) {
  const currentIndex = getStepIndex(currentStep);
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-6 px-1 pb-6 sm:px-2">
      <div className="flex items-center gap-2 pr-12">
        <Text variant="lead-medium" as="span" className="text-textBlack">
          Publish agent
        </Text>
      </div>

      <ol
        aria-label="Publish progress"
        className="flex w-full items-center gap-3"
      >
        {steps.map((step, index) => {
          const isComplete = index < currentIndex;
          const isCurrent = index === currentIndex;
          const isLast = index === steps.length - 1;

          return (
            <Fragment key={step.key}>
              <li
                aria-current={isCurrent ? "step" : undefined}
                className="flex shrink-0 items-center gap-2"
              >
                <motion.span
                  animate={
                    shouldReduceMotion
                      ? undefined
                      : { scale: isCurrent ? 1.04 : 1 }
                  }
                  transition={
                    shouldReduceMotion
                      ? { duration: 0 }
                      : { duration: 0.22, ease: [0.16, 1, 0.3, 1] }
                  }
                  className={cn(
                    "flex size-6 shrink-0 items-center justify-center rounded-full text-[12px] font-medium transition-colors",
                    isCurrent
                      ? "bg-zinc-900 text-white"
                      : isComplete
                        ? "bg-zinc-900 text-white"
                        : "border border-zinc-300 bg-white text-zinc-400",
                  )}
                >
                  <AnimatePresence mode="wait" initial={false}>
                    {isComplete ? (
                      <motion.span
                        key="check"
                        initial={
                          shouldReduceMotion
                            ? { opacity: 0 }
                            : { opacity: 0, scale: 0.6 }
                        }
                        animate={{ opacity: 1, scale: 1 }}
                        exit={
                          shouldReduceMotion
                            ? { opacity: 0 }
                            : { opacity: 0, scale: 0.6 }
                        }
                        transition={{ duration: 0.2, ease: "easeOut" }}
                        className="flex items-center justify-center"
                      >
                        <CheckIcon size={12} weight="bold" />
                      </motion.span>
                    ) : (
                      <motion.span
                        key="num"
                        initial={
                          shouldReduceMotion
                            ? { opacity: 0 }
                            : { opacity: 0, y: 4 }
                        }
                        animate={{ opacity: 1, y: 0 }}
                        exit={
                          shouldReduceMotion
                            ? { opacity: 0 }
                            : { opacity: 0, y: -4 }
                        }
                        transition={{ duration: 0.18, ease: "easeOut" }}
                      >
                        {index + 1}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.span>
                <Text
                  variant="small-medium"
                  as="span"
                  className={cn(
                    "whitespace-nowrap !text-current transition-colors",
                    isCurrent
                      ? "text-zinc-950"
                      : isComplete
                        ? "text-zinc-700"
                        : "font-normal text-zinc-400",
                  )}
                >
                  {step.label}
                </Text>
              </li>
              {!isLast ? (
                <li aria-hidden className="relative h-0 flex-1">
                  <span className="absolute inset-x-0 top-0 border-t border-solid border-zinc-200" />
                  <motion.span
                    initial={false}
                    animate={{ scaleX: isComplete ? 1 : 0 }}
                    style={{ transformOrigin: "left" }}
                    transition={
                      shouldReduceMotion
                        ? { duration: 0 }
                        : { duration: 0.32, ease: [0.16, 1, 0.3, 1] }
                    }
                    className="absolute inset-x-0 top-0 border-t border-solid border-zinc-900"
                  />
                </li>
              ) : null}
            </Fragment>
          );
        })}
      </ol>
    </div>
  );
}
