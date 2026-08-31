import {
  getGetExpertQueryKey,
  useHireExpert,
  useListExperts,
  useUpdateExpertSoul,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { HireResult } from "@/app/api/__generated__/models/hireResult";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  buildVoicePreferences,
  type VoicePickResult,
} from "@/components/organisms/VoicePicker/helpers";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

function celebrate(result: HireResult) {
  toast({
    title: `${result.expert.name} joined your team`,
    description: result.failed_preloads.length
      ? `Couldn't attach: ${result.failed_preloads.join(", ")}`
      : undefined,
    variant: "success",
    action: (
      <Button as="NextLink" href="/team" variant="ghost" size="small">
        View team
      </Button>
    ),
  });
}

export function useExpertProfileSheet(
  expert: Expert | null,
  onClose: () => void,
) {
  const queryClient = useQueryClient();
  const router = useRouter();
  // Set once a hire succeeds for a persona that ships writing samples: it
  // swaps the sheet to the voice pick before the hire is celebrated.
  const [hireResult, setHireResult] = useState<HireResult | null>(null);
  const pendingCelebrationRef = useRef<HireResult | null>(null);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[] },
  });

  const isHired =
    expert !== null &&
    (!expert.is_template ||
      (expertsQuery.data ?? []).some(
        (hired) => hired.source_template_id === expert.id,
      ));

  const { mutateAsync: hireExpert, isPending: isHiring } = useHireExpert();
  const { mutateAsync: updateSoul, isPending: isSavingVoice } =
    useUpdateExpertSoul();

  function handleClose() {
    const completedHire = pendingCelebrationRef.current;
    pendingCelebrationRef.current = null;
    setHireResult(null);
    if (completedHire) {
      celebrate(completedHire);
      // Hiring isn't installing: hand the user straight to the expert's thread
      // with kickoff=1 so it introduces itself and starts its day-one job.
      router.push(`/copilot?expertId=${completedHire.expert.id}&kickoff=1`);
    }
    onClose();
  }

  async function hire() {
    if (!expert || !expert.is_template) return;
    try {
      const response = await hireExpert({ data: { template_id: expert.id } });
      const result = response.data as HireResult;
      await invalidateExpertRosterQueries(queryClient);
      pendingCelebrationRef.current = result;
      // Offer the voice pick when the persona ships writing samples; otherwise
      // finish with the classic celebration toast.
      if ((expert.voice_samples ?? []).length > 0) {
        setHireResult(result);
        return;
      }
      handleClose();
    } catch {
      toast({
        title: `Couldn't hire ${expert.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function pickVoice(result: VoicePickResult) {
    if (!hireResult || !expert) return;
    const hired = hireResult.expert;
    const voicePreferences = buildVoicePreferences(
      result,
      expert.voice_samples ?? [],
    );
    // Nothing storable (missing sample, blank custom text): keep the hired
    // expert's existing voice instead of blanking it with an empty PATCH.
    if (voicePreferences === null) {
      skipVoice();
      return;
    }
    try {
      await updateSoul({
        expertId: hired.id,
        data: {
          name: hired.name,
          identity: hired.identity,
          voice_preferences: voicePreferences,
          boundaries: hired.boundaries,
        },
      });
      // The hire-time refetch cached the pre-voice expert as fresh; without
      // this, a Soul edit within the stale window would silently write that
      // old description back over the chosen voice.
      await Promise.all([
        invalidateExpertRosterQueries(queryClient),
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(hired.id),
        }),
      ]);
    } catch {
      // The hire itself succeeded; keep the picker open so the choice can be
      // retried, and point at the Soul editor as the fallback.
      toast({
        title: "Couldn't save the voice",
        description: "You can set it later in the Soul editor.",
        variant: "destructive",
      });
      return;
    }
    handleClose();
  }

  function skipVoice() {
    handleClose();
  }

  return {
    isHired,
    isHiring,
    hire,
    hireResult,
    pickVoice,
    skipVoice,
    isSavingVoice,
    handleClose,
  };
}
