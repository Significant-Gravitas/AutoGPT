"use client";

import * as React from "react";

import Image from "next/image";
import { motion, useReducedMotion } from "framer-motion";
import { StepFooter } from "./StepFooter";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Confetti } from "@/components/molecules/Confetti/Confetti";
import { usePathname } from "next/navigation";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import {
  ArrowRightIcon,
  CheckIcon,
  ClockIcon,
  ImageBrokenIcon,
  PaperPlaneTiltIcon,
  RocketLaunchIcon,
  WarningCircleIcon,
  XIcon,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

interface Props {
  agentName: string;
  subheader: string;
  description: string;
  onClose: () => void;
  onDone: () => void;
  onViewProgress: () => void;
  thumbnailSrc?: string;
  status?: SubmissionStatus;
  reviewComments?: string | null;
}

function getHeroContent(
  status: SubmissionStatus | undefined,
  isDashboardPage: boolean,
) {
  switch (status) {
    case SubmissionStatus.APPROVED:
      return {
        title: "Agent approved",
        description:
          "Your agent has been approved and is now live on the AutoGPT marketplace.",
        Icon: CheckIcon,
        ringClass: "bg-emerald-50 ring-emerald-100",
        iconClass: "bg-emerald-500 text-white",
      };
    case SubmissionStatus.REJECTED:
      return {
        title: "Agent needs changes",
        description:
          "Your submission was not approved. Review the feedback and resubmit.",
        Icon: XIcon,
        ringClass: "bg-rose-50 ring-rose-100",
        iconClass: "bg-rose-500 text-white",
      };
    default:
      return {
        title: "Submission received",
        description: isDashboardPage
          ? "We'll notify you once review is complete. Approved agents go live on the marketplace."
          : "We'll notify you once review is complete. Track progress from the Creator Dashboard.",
        Icon: CheckIcon,
        ringClass: "bg-purple-50 ring-purple-100",
        iconClass: "bg-purple-500 text-white",
      };
  }
}

export function AgentReviewStep({
  agentName,
  subheader,
  description: _description,
  thumbnailSrc,
  onDone,
  onViewProgress,
  status,
  reviewComments,
}: Props) {
  const pathname = usePathname();
  const isDashboardPage = pathname.includes("/settings/creator-dashboard");
  const hero = getHeroContent(status, isDashboardPage);
  const HeroIcon = hero.Icon;
  const shouldReduceMotion = useReducedMotion();
  const isPending =
    status !== SubmissionStatus.APPROVED &&
    status !== SubmissionStatus.REJECTED;

  const showCelebration = status !== SubmissionStatus.REJECTED;
  const showConfetti = showCelebration && !shouldReduceMotion;

  return (
    <div
      aria-labelledby="modal-title"
      className="relative flex flex-col items-center pb-4 pt-10"
    >
      {showConfetti ? (
        <Confetti
          options={{
            particleCount: 80,
            spread: 70,
            startVelocity: 35,
            origin: { y: 0.3 },
          }}
        />
      ) : null}

      <div className="relative flex items-center justify-center">
        {showCelebration && !shouldReduceMotion ? (
          <>
            <motion.span
              aria-hidden
              initial={{ opacity: 0.5, scale: 0.6 }}
              animate={{ opacity: 0, scale: 1.6 }}
              transition={{
                duration: 1.6,
                ease: "easeOut",
                repeat: Infinity,
                repeatDelay: 0.4,
              }}
              className={cn(
                "absolute inline-block size-24 rounded-full",
                hero.iconClass.includes("bg-emerald")
                  ? "bg-emerald-400/40"
                  : hero.iconClass.includes("bg-rose")
                    ? "bg-rose-400/40"
                    : "bg-purple-400/40",
              )}
            />
            <motion.span
              aria-hidden
              initial={{ opacity: 0.4, scale: 0.7 }}
              animate={{ opacity: 0, scale: 1.4 }}
              transition={{
                duration: 1.6,
                ease: "easeOut",
                repeat: Infinity,
                repeatDelay: 0.4,
                delay: 0.5,
              }}
              className={cn(
                "absolute inline-block size-24 rounded-full",
                hero.iconClass.includes("bg-emerald")
                  ? "bg-emerald-400/40"
                  : hero.iconClass.includes("bg-rose")
                    ? "bg-rose-400/40"
                    : "bg-purple-400/40",
              )}
            />
          </>
        ) : null}
        <motion.div
          initial={
            shouldReduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.7 }
          }
          animate={{ opacity: 1, scale: 1 }}
          transition={
            shouldReduceMotion
              ? { duration: 0 }
              : { type: "spring", stiffness: 280, damping: 20, delay: 0.05 }
          }
          className={cn(
            "relative flex size-20 items-center justify-center rounded-full shadow-[0_8px_24px_-8px_rgba(119,51,245,0.45)]",
            hero.iconClass.includes("bg-emerald")
              ? "bg-gradient-to-br from-emerald-400 to-emerald-600"
              : hero.iconClass.includes("bg-rose")
                ? "bg-gradient-to-br from-rose-400 to-rose-600"
                : "bg-gradient-to-br from-purple-400 to-purple-600",
          )}
        >
          <span
            aria-hidden
            className="absolute inset-1 rounded-full bg-white/10"
          />
          <motion.span
            initial={
              shouldReduceMotion ? { opacity: 0 } : { scale: 0.4, opacity: 0 }
            }
            animate={{ scale: 1, opacity: 1 }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : {
                    type: "spring",
                    stiffness: 360,
                    damping: 18,
                    delay: 0.18,
                  }
            }
            className="text-white"
          >
            <HeroIcon size={36} weight="bold" />
          </motion.span>
        </motion.div>
      </div>

      <motion.div
        initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut", delay: 0.12 }}
        className="mt-5 flex max-w-md flex-col items-center gap-2 px-2 text-center"
      >
        <Text
          variant="lead-medium"
          as="h2"
          className="text-textBlack"
          data-testid="view-agent-name"
        >
          {hero.title}
        </Text>
        <Text variant="body" className="text-zinc-600">
          {hero.description}
        </Text>
      </motion.div>

      <motion.div
        initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut", delay: 0.18 }}
        className="mt-6 flex w-full max-w-md items-center gap-3 rounded-[14px] border border-zinc-200 bg-white p-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]"
      >
        <div className="relative aspect-video h-12 shrink-0 overflow-hidden rounded-[8px] bg-zinc-100">
          {thumbnailSrc ? (
            <Image
              src={thumbnailSrc}
              alt=""
              fill
              sizes="86px"
              className="object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-zinc-400">
              <ImageBrokenIcon size={20} weight="duotone" />
            </div>
          )}
        </div>
        <div className="flex min-w-0 flex-1 flex-col">
          <Text
            variant="body-medium"
            as="span"
            className="truncate text-textBlack"
          >
            {agentName}
          </Text>
          {subheader ? (
            <Text variant="small" className="truncate text-zinc-500">
              {subheader}
            </Text>
          ) : null}
        </div>
        {isPending ? (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-medium text-amber-800">
            <ClockIcon size={12} weight="duotone" />
            In review
          </span>
        ) : null}
      </motion.div>

      {reviewComments && status === SubmissionStatus.REJECTED ? (
        <div className="mt-4 w-full max-w-md rounded-[14px] border border-rose-200 bg-rose-50 p-3">
          <div className="mb-1 flex items-center gap-2 text-rose-700">
            <WarningCircleIcon size={16} weight="duotone" />
            <Text variant="small-medium" as="span" className="!text-current">
              Review feedback
            </Text>
          </div>
          <Text variant="small" className="text-rose-700">
            {reviewComments}
          </Text>
        </div>
      ) : null}

      {status !== SubmissionStatus.REJECTED ? (
        <motion.div
          initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.24, ease: "easeOut", delay: 0.24 }}
          className="mt-6 flex w-full max-w-md flex-col gap-3 px-2"
        >
          <Text variant="body-medium" as="span" className="text-textBlack">
            What happens next
          </Text>
          <ol className="flex flex-col gap-3">
            <li className="flex items-start gap-3">
              <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-purple-50 text-purple-600">
                <PaperPlaneTiltIcon size={14} weight="bold" />
              </span>
              <div className="flex min-w-0 flex-col">
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-textBlack"
                >
                  Submitted for review
                </Text>
                <Text variant="small" className="text-zinc-500">
                  Your listing is queued in the marketplace review pipeline.
                </Text>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-amber-50 text-amber-700">
                <ClockIcon size={14} weight="bold" />
              </span>
              <div className="flex min-w-0 flex-col">
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-textBlack"
                >
                  Reviewed soon
                </Text>
                <Text variant="small" className="text-zinc-500">
                  Our team checks the details, media, and safety of your agent.
                </Text>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-emerald-50 text-emerald-700">
                <RocketLaunchIcon size={14} weight="bold" />
              </span>
              <div className="flex min-w-0 flex-col">
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-textBlack"
                >
                  Approved listings go live
                </Text>
                <Text variant="small" className="text-zinc-500">
                  You&apos;ll get an email; rejected listings come back with
                  feedback.
                </Text>
              </div>
            </li>
          </ol>
        </motion.div>
      ) : null}

      <div className="mt-8 w-full">
        <StepFooter
          secondary={
            <Button
              variant="secondary"
              size="small"
              onClick={onDone}
              className="w-full sm:w-auto"
            >
              Done
            </Button>
          }
          primary={
            <Button
              size="small"
              onClick={onViewProgress}
              className="w-full sm:w-auto"
              rightIcon={<ArrowRightIcon size={14} weight="bold" />}
              data-testid="view-progress-button"
            >
              {isDashboardPage ? "View progress" : "Go to Creator Dashboard"}
            </Button>
          }
        />
      </div>
    </div>
  );
}
