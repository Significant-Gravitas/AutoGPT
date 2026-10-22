import { useHireExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { HIRE_ALL_ADVANCE_DELAY_MS, hireFailureToast } from "./helpers";

interface Args {
  experts: RecommendedExpert[];
  onTeamHired: () => void;
}

// The step's primary action: hires a batch in parallel, lets each card flip
// as its own request lands, and moves on only when every hire succeeded.
export function useHireAll({ experts, onTeamHired }: Args) {
  const markHired = useOnboardingWizardStore((s) => s.markHired);
  const setStepBusy = useOnboardingWizardStore((s) => s.setStepBusy);
  const queryClient = useQueryClient();
  const { mutateAsync: hireExpert } = useHireExpert();
  const [isHiringAll, setIsHiringAll] = useState(false);
  const [hiringTemplateIds, setHiringTemplateIds] = useState<string[]>([]);
  const [failedTemplateIds, setFailedTemplateIds] = useState<string[]>([]);
  const advanceTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const isMounted = useRef(false);

  // Busy through the advance beat too: Back then would land the timer's
  // nextStep on the previous step.
  useEffect(() => {
    setStepBusy(isHiringAll);
    return () => setStepBusy(false);
  }, [isHiringAll, setStepBusy]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      clearTimeout(advanceTimer.current);
    };
  }, []);

  async function hireAll(batch: RecommendedExpert[], isRetry: boolean) {
    trackBrainDump("hire_all_clicked", {
      count: batch.length,
      already_hired: hiredCount(experts),
      recommended: experts.length,
      retry: isRetry,
    });
    setIsHiringAll(true);
    setFailedTemplateIds([]);
    setHiringTemplateIds(batch.map((expert) => expert.template_id));

    const results = await Promise.allSettled(batch.map(hireOne));
    const failed = batch.filter((_, i) => results[i].status === "rejected");
    // Best effort, as for a card hire: the roster is not on screen here.
    await invalidateExpertRosterQueries(queryClient).catch(() => undefined);
    // Browser navigation can unmount the step mid-batch; the wizard store
    // outlives it, so a stale batch must not drive the wizard.
    if (!isMounted.current) return;

    if (failed.length === 0) {
      advanceTimer.current = setTimeout(onTeamHired, HIRE_ALL_ADVANCE_DELAY_MS);
      return;
    }
    setFailedTemplateIds(failed.map((expert) => expert.template_id));
    setIsHiringAll(false);
    toast({
      ...hireFailureToast(failed, batch.length, hiredCount(experts) > 0),
      variant: "destructive",
    });
  }

  async function hireOne(expert: RecommendedExpert) {
    trackFunnel("hire_started", {
      template_id: expert.template_id,
      surface: "onboarding",
    });
    try {
      await hireExpert({
        data: { template_id: expert.template_id, surface: "onboarding" },
      });
      markHired(expert.template_id);
    } finally {
      setHiringTemplateIds((ids) =>
        ids.filter((id) => id !== expert.template_id),
      );
    }
  }

  return { isHiringAll, hiringTemplateIds, failedTemplateIds, hireAll };
}

// Read from the store, not the render: the batch's own hires land after the
// closure was made.
function hiredCount(experts: RecommendedExpert[]) {
  const hired = useOnboardingWizardStore.getState().hiredTemplateIds;
  return experts.filter((expert) => hired.includes(expert.template_id)).length;
}
