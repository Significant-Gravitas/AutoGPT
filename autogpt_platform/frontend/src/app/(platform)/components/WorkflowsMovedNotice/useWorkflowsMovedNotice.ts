"use client";

import {
  getGetV1OnboardingStateQueryKey,
  useGetV1OnboardingState,
  usePostV1CompleteOnboardingStep,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useNoticeAdmission } from "./useNoticeAdmission";
import {
  isPreExpertsUser,
  hasPendingChatHandoff,
  isWorkflowsMovedNoticeRoute,
  peekWorkflowsMovedNoticeSeen,
  setWorkflowsMovedNoticeSeen,
} from "./helpers";

const STEP = PostV1CompleteOnboardingStepStep.WORKFLOWS_MOVED;

export function useWorkflowsMovedNotice(canShow = true) {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const userID = user?.id ?? null;
  const isSafeLanding = useNoticeLandingRoute();
  const experts = useFlagStatus(Flag.HIRE_EXPERTS);
  const layout = useFlagStatus(Flag.AUTOGPT_NEW_LAYOUT);
  const [dismissedUsers, setDismissedUsers] = useState<string[]>([]);
  const isEligible = Boolean(
    canShow &&
      userID &&
      experts.ready &&
      experts.enabled &&
      layout.ready &&
      layout.enabled &&
      isPreExpertsUser(user?.created_at) &&
      isSafeLanding &&
      !dismissedUsers.includes(userID) &&
      !peekWorkflowsMovedNoticeSeen(userID),
  );
  const { data, queryKey } = useGetV1OnboardingState({
    query: {
      queryKey: [
        ...getGetV1OnboardingStateQueryKey(),
        "workflows-moved",
        userID,
      ],
      enabled: isEligible,
      staleTime: 60_000,
    },
  });
  const { mutate: completeStep } = usePostV1CompleteOnboardingStep();
  const steps = data?.status === 200 ? data.data.completedSteps : [];
  const isReadyToShow =
    isEligible &&
    steps.includes("ONBOARDING_COMPLETE") &&
    !steps.includes(STEP);
  const isAdmitted = useNoticeAdmission(isReadyToShow, userID);
  const hasStartedWorking = useNoticeInteraction(isSafeLanding, isAdmitted);
  const isOpen = isReadyToShow && isAdmitted && !hasStartedWorking;

  function dismiss() {
    if (!isOpen || !userID || data?.status !== 200) return;
    setDismissedUsers((users) => [...users, userID]);
    setWorkflowsMovedNoticeSeen(userID);
    queryClient.setQueryData(queryKey, {
      ...data,
      data: {
        ...data.data,
        completedSteps: [...data.data.completedSteps, STEP],
      },
    });
    completeStep({ params: { step: STEP } });
  }

  return { isOpen, dismiss };
}

function useNoticeLandingRoute() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const isCopilot = pathname?.replace(/\/$/, "") === "/copilot";
  const isSafeRoute = isWorkflowsMovedNoticeRoute(pathname, searchParams);
  const hasChatIntent = isCopilot && (!isSafeRoute || hasPendingChatHandoff());
  const [deferCopilotVisit, setDeferCopilotVisit] = useState(hasChatIntent);

  useEffect(() => {
    if (!isCopilot) setDeferCopilotVisit(false);
    else if (hasChatIntent) setDeferCopilotVisit(true);
  }, [hasChatIntent, isCopilot]);

  return isSafeRoute && !hasChatIntent && !(isCopilot && deferCopilotVisit);
}

function useNoticeInteraction(isSafeLanding: boolean, isReadyToShow: boolean) {
  const pathname = usePathname();
  const [deferredPath, setDeferredPath] = useState<string | null>(null);

  useEffect(() => setDeferredPath(null), [pathname]);

  useEffect(() => {
    if (!isSafeLanding || isReadyToShow) return;

    function deferThisVisit() {
      setDeferredPath(pathname);
    }

    document.addEventListener("pointerdown", deferThisVisit, true);
    document.addEventListener("keydown", deferThisVisit, true);
    return () => {
      document.removeEventListener("pointerdown", deferThisVisit, true);
      document.removeEventListener("keydown", deferThisVisit, true);
    };
  }, [pathname, isSafeLanding, isReadyToShow]);

  return deferredPath !== null && deferredPath === pathname;
}
