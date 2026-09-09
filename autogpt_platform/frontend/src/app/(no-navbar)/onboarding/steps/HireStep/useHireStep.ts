import { useGetBrainDumpRecommendedExperts } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useHireExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import {
  RECOMMENDATIONS_MAX_WAIT_MS,
  RECOMMENDATIONS_POLL_MS,
} from "./helpers";

export function useHireStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const hiredTemplateIds = useOnboardingWizardStore((s) => s.hiredTemplateIds);
  const markHired = useOnboardingWizardStore((s) => s.markHired);
  const queryClient = useQueryClient();
  const [hiringTemplateId, setHiringTemplateId] = useState<string | null>(null);
  const [gaveUpWaiting, setGaveUpWaiting] = useState(false);
  const reportedIds = useRef(new Set<string>());

  const { data } = useGetBrainDumpRecommendedExperts({
    query: {
      refetchInterval: (query) => {
        if (gaveUpWaiting) return false;
        const response = query.state.data;
        if (response && (response.status !== 200 || response.data.ready)) {
          return false;
        }
        return RECOMMENDATIONS_POLL_MS;
      },
    },
  });

  const answered = data !== undefined;
  const isReadyPerServer = data?.status !== 200 || data.data.ready;
  const isPending = !gaveUpWaiting && (!answered || !isReadyPerServer);
  const team: ExpertRecommendations | null =
    data?.status === 200 ? (data.data.team ?? null) : null;

  useEffect(() => {
    if (!isPending) return;
    const timer = setTimeout(
      () => setGaveUpWaiting(true),
      RECOMMENDATIONS_MAX_WAIT_MS,
    );
    return () => clearTimeout(timer);
  }, [isPending]);

  // A poll hands back a new object for the same team; the impression for
  // each card is reported once, not once per poll.
  useEffect(() => {
    if (isPending) return;
    (team?.experts ?? []).forEach((expert, position) => {
      if (reportedIds.current.has(expert.template_id)) return;
      reportedIds.current.add(expert.template_id);
      trackBrainDump("expert_recommended", {
        template_id: expert.template_id,
        position,
        source: team?.source ?? null,
      });
    });
  }, [team, isPending]);

  const { mutateAsync: hireExpert } = useHireExpert();

  async function hire(expert: RecommendedExpert, position: number) {
    if (hiringTemplateId !== null) return;
    trackBrainDump("expert_recommendation_clicked", {
      template_id: expert.template_id,
      position,
    });
    trackBrainDump("hire_started", {
      template_id: expert.template_id,
      source: "onboarding_hire_step",
    });
    setHiringTemplateId(expert.template_id);
    try {
      await hireExpert({ data: { template_id: expert.template_id } });
      markHired(expert.template_id);
      trackBrainDump("onboarding_expert_hired", {
        template_id: expert.template_id,
        position,
      });
      // Best effort: the roster is not on screen here, and a failed
      // invalidation must not read as a failed hire.
      await invalidateExpertRosterQueries(queryClient).catch(() => undefined);
    } catch {
      toast({
        title: `Couldn't hire ${expert.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setHiringTemplateId(null);
    }
  }

  function handleContinue() {
    trackBrainDump("hire_step_continued", {
      hired: hiredTemplateIds.length,
      recommended: team?.experts?.length ?? 0,
    });
    nextStep();
  }

  return {
    team,
    isPending,
    experts: team?.experts ?? [],
    raiseRole: team?.raise_suggestion?.role ?? null,
    hiredTemplateIds,
    hiringTemplateId,
    hire,
    handleContinue,
  };
}
