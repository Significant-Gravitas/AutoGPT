import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { useToast } from "@/components/useToast";
import { chatKeys } from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";
import {
  completeOnboarding,
  finalizeBrainDump,
  getOnboardingPlan,
  hireOnboardingPlan,
  submitOnboardingProfile,
} from "@/lib/api/onboarding";
import { signOut } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import {
  finishOnboarding,
  isPlanRequired,
  joinNames,
  type FinishStage,
  type OnboardingAPI,
} from "./finishOnboarding";
import type { WorkAnswer } from "./helpers";
import { markOnboarded } from "./onboardedStore";

const SETUP_FAILED =
  "We couldn't finish setting up your account. Your answers are saved; try again.";
// Long enough to read the toast before Home replaces this screen.
const NOT_HIRED_PAUSE_MS = 4_000;

const api: OnboardingAPI = {
  submitProfile: submitOnboardingProfile,
  finalizeDump: finalizeBrainDump,
  async isPaywalled() {
    try {
      await listExperts();
      return false;
    } catch (error) {
      // Any other failure is not a paywall; the hire then says if it is one.
      return isPlanRequired(error);
    }
  },
  getPlan: getOnboardingPlan,
  hirePlan: hireOnboardingPlan,
  complete: completeOnboarding,
};

interface Args {
  userID: string;
  userName: string;
  role: string;
  work: WorkAnswer;
  recordingID: string;
  onPlanNeeded: () => void;
}

export function useSetupStep({
  userID,
  userName,
  role,
  work,
  recordingID,
  onPlanNeeded,
}: Args) {
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const [stage, setStage] = useState<FinishStage>("saving");
  const [teamNames, setTeamNames] = useState<string[]>([]);
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isRunning = useRef(false);

  async function run() {
    if (isRunning.current) return;
    isRunning.current = true;
    setError(null);
    try {
      const result = await finishOnboarding({
        api,
        userName,
        role,
        work,
        recordingID,
        onStage(next, names) {
          setStage(next);
          setTeamNames(names);
        },
      });
      if (result.kind === "planNeeded") {
        onPlanNeeded();
        return;
      }
      setIsDone(true);
      void queryClient.invalidateQueries({ queryKey: expertKeys.list });
      void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
      if (result.notHired.length > 0) {
        show({
          tone: "error",
          message: `Couldn't hire ${joinNames(result.notHired)}. Otto takes on their tasks for now. You can hire experts any time.`,
        });
        await new Promise((resolve) => setTimeout(resolve, NOT_HIRED_PAUSE_MS));
      }
      // Swaps the root stack to Home.
      markOnboarded(userID);
    } catch {
      setError(SETUP_FAILED);
    } finally {
      isRunning.current = false;
    }
  }

  useMountEffect(() => {
    void run();
  });

  return {
    stage,
    teamNames,
    isDone,
    error,
    toast,
    retry: run,
    // A setup that keeps failing must not trap the person on this screen.
    signOut,
  };
}
