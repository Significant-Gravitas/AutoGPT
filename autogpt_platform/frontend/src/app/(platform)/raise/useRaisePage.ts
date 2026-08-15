import {
  getListExpertsQueryKey,
  useCreateRaisedExpert,
  type listExpertsResponse,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { useRef, useState } from "react";
import {
  buildTranscript,
  clearDraft,
  EMPTY_DRAFT,
  getFirstJobFailureToast,
  getRaiseErrorToast,
  loadDraft,
  previousStep,
  reconcileCreatedExpert,
  resolveVoicePreferences,
  saveDraft,
  voiceSummaryLabel,
  VOICE_SAMPLES,
  type RaiseDraft,
} from "./helpers";

export function useRaisePage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const isNaming = searchParams.get("from") === "naming";
  const { mutateAsync: createRaisedExpert, isPending } =
    useCreateRaisedExpert();
  const [draft, setDraft] = useState<RaiseDraft>(() =>
    isNaming ? EMPTY_DRAFT : loadDraft(),
  );
  const [isSubmissionLocked, setIsSubmissionLocked] = useState(false);
  const afterVoiceStep = isNaming ? "review" : "firstJob";
  // Synchronous latch: isPending only flips after a rerender, so a rapid
  // double-click could dispatch two POSTs without it. Stays latched after
  // success (we are navigating away); resets only on error so the user
  // can retry.
  const submitLatch = useRef(false);

  function update(changes: Partial<RaiseDraft>) {
    const next = { ...draft, ...changes };
    saveDraft(next);
    setDraft(next);
  }

  function submitName(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    update({ name: trimmed, step: "voice" });
  }

  function pickVoice(result: VoicePickResult) {
    const preferences = resolveVoicePreferences(result, VOICE_SAMPLES);
    if (preferences === null) {
      skipVoice();
      return;
    }
    update({
      voicePreferences: preferences,
      voiceLabel: voiceSummaryLabel(result, VOICE_SAMPLES),
      step: afterVoiceStep,
    });
  }

  function skipVoice() {
    update({
      voicePreferences: "",
      voiceLabel: null,
      step: afterVoiceStep,
    });
  }

  function pickFirstJob(job: { id: string; name: string }) {
    update({ firstJob: job, step: "review" });
  }

  function skipFirstJob() {
    update({ firstJob: null, step: "review" });
  }

  function goBack() {
    update({ step: previousStep(draft.step, isNaming) });
  }

  async function finish() {
    if (submitLatch.current) return;
    submitLatch.current = true;
    setIsSubmissionLocked(true);
    try {
      const response = await createRaisedExpert({
        data: {
          name: draft.name,
          voice_preferences: draft.voicePreferences || null,
          first_job_store_listing_version_id: draft.firstJob?.id ?? null,
        },
      });
      if (response.status !== 200) {
        throw new Error(`Unexpected raise response: ${response.status}`);
      }
      const result = response.data;
      clearDraft();
      const firstJobToast = getFirstJobFailureToast(result, draft.firstJob);
      if (firstJobToast) toast(firstJobToast);
      queryClient.setQueryData<listExpertsResponse>(
        getListExpertsQueryKey(),
        (cached) => reconcileCreatedExpert(cached, result.expert),
      );
      const kickoff = result.expert.workflows.length > 0 ? "&kickoff=1" : "";
      router.push(`/copilot?expertId=${result.expert.id}${kickoff}`);
    } catch (error) {
      submitLatch.current = false;
      setIsSubmissionLocked(false);
      toast(getRaiseErrorToast(error, draft.name));
    }
  }

  return {
    step: draft.step,
    messages: buildTranscript(draft, isNaming),
    name: draft.name,
    voiceLabel: draft.voiceLabel,
    firstJob: draft.firstJob,
    isSubmitting: isPending || isSubmissionLocked,
    submitName,
    pickVoice,
    skipVoice,
    pickFirstJob,
    skipFirstJob,
    goBack,
    finish,
  };
}
