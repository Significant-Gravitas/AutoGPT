"use client";

import { usePathname } from "next/navigation";
import { useReducedMotion } from "framer-motion";
import {
  CheckIcon,
  NotePencilIcon,
  XIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { getSubmissionMeta } from "../helpers";

interface HeroContent {
  title: string;
  description: string;
  Icon: PhosphorIcon;
  pulse: string;
  gradient: string;
}

function getHeroContent(
  status: SubmissionStatus | undefined,
  isDashboardPage: boolean,
): HeroContent {
  switch (status) {
    case SubmissionStatus.APPROVED:
      return {
        title: "Agent approved",
        description:
          "Your agent has been approved and is now live on the AutoGPT marketplace.",
        Icon: CheckIcon,
        pulse: "bg-emerald-400/40",
        gradient: "from-emerald-400 to-emerald-600",
      };
    case SubmissionStatus.REJECTED:
      return {
        title: "Agent needs changes",
        description:
          "Your submission was not approved. Review the feedback and resubmit.",
        Icon: XIcon,
        pulse: "bg-rose-400/40",
        gradient: "from-rose-400 to-rose-600",
      };
    case SubmissionStatus.DRAFT:
      return {
        title: "Draft saved",
        description:
          "This agent isn't submitted yet. Finish the details and submit it for review.",
        Icon: NotePencilIcon,
        pulse: "bg-zinc-400/40",
        gradient: "from-zinc-400 to-zinc-600",
      };
    default:
      return {
        title: "Submission received",
        description: isDashboardPage
          ? "We'll notify you once review is complete. Approved agents go live on the marketplace."
          : "We'll notify you once review is complete. Track progress from the Creator Dashboard.",
        Icon: CheckIcon,
        pulse: "bg-purple-400/40",
        gradient: "from-purple-400 to-purple-600",
      };
  }
}

interface Args {
  status: SubmissionStatus | undefined;
  version: number | undefined;
  category: string | null | undefined;
  submittedAt: string | Date | null | undefined;
  reviewedAt: string | Date | null | undefined;
  runCount: number | undefined;
}

export function useAgentReviewStep({
  status,
  version,
  category,
  submittedAt,
  reviewedAt,
  runCount,
}: Args) {
  const pathname = usePathname();
  const isDashboardPage = pathname.includes("/settings/creator-dashboard");
  const hero = getHeroContent(status, isDashboardPage);
  const shouldReduceMotion = useReducedMotion();

  const isApproved = status === SubmissionStatus.APPROVED;
  const isRejected = status === SubmissionStatus.REJECTED;
  const isDraft = status === SubmissionStatus.DRAFT;
  const isPending = !isApproved && !isRejected && !isDraft;

  const showCelebration = isApproved || (isPending && !isDashboardPage);
  const showConfetti = showCelebration && !shouldReduceMotion;

  const metaItems = getSubmissionMeta({
    status,
    version,
    category,
    submittedAt,
    reviewedAt,
    runCount,
  });

  return {
    isDashboardPage,
    hero,
    shouldReduceMotion,
    isApproved,
    isRejected,
    isDraft,
    isPending,
    showCelebration,
    showConfetti,
    metaItems,
  };
}
