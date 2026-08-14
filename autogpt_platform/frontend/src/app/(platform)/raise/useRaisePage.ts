import { useCreateRaisedExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import {
  buildTranscript,
  clearDraft,
  loadDraft,
  previousStep,
  resolveVoicePreferences,
  saveDraft,
  voiceSummaryLabel,
  VOICE_SAMPLES,
  type RaiseDraft,
} from "./helpers";

export function useRaisePage() {
  const router = useRouter();
  const { mutateAsync: createRaisedExpert, isPending } =
    useCreateRaisedExpert();
  const [draft, setDraft] = useState<RaiseDraft>(loadDraft);
  // Synchronous latch: isPending only flips after a rerender, so a rapid
  // double-click could dispatch two POSTs without it. Stays latched after
  // success (we are navigating away); resets only on error so the user
  // can retry.
  const submitLatch = useRef(false);

  function update(changes: Partial<RaiseDraft>) {
    setDraft((prev) => {
      const next = { ...prev, ...changes };
      saveDraft(next);
      return next;
    });
  }

  function submitName(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    update({ name: trimmed, step: "voice" });
  }

  function pickVoice(result: VoicePickResult) {
    const preferences = resolveVoicePreferences(result);
    if (preferences === null) {
      skipVoice();
      return;
    }
    update({
      voicePreferences: preferences,
      voiceLabel: voiceSummaryLabel(result, VOICE_SAMPLES),
      voiceSkipped: false,
      step: "firstJob",
    });
  }

  function skipVoice() {
    update({
      voicePreferences: "",
      voiceLabel: null,
      voiceSkipped: true,
      step: "firstJob",
    });
  }

  function pickFirstJob(job: { id: string; name: string }) {
    update({ firstJob: job, firstJobSkipped: false, step: "review" });
  }

  function skipFirstJob() {
    update({ firstJob: null, firstJobSkipped: true, step: "review" });
  }

  function goBack() {
    update({ step: previousStep(draft.step) });
  }

  async function finish() {
    if (submitLatch.current) return;
    submitLatch.current = true;
    try {
      const response = await createRaisedExpert({
        data: {
          name: draft.name,
          voice_preferences: draft.voicePreferences || null,
          first_job_store_listing_version_id: draft.firstJob?.id ?? null,
        },
      });
      const result = response.data as RaiseResult;
      clearDraft();
      if (draft.firstJob && !result.first_job_installed) {
        toast({
          title: `Couldn't set up ${result.expert.name}'s first job`,
          description: `You can install "${draft.firstJob.name}" from her page anytime.`,
          variant: "default",
        });
      }
      const kickoff = result.expert.workflows.length > 0 ? "&kickoff=1" : "";
      router.push(`/copilot?expertId=${result.expert.id}${kickoff}`);
    } catch (error) {
      submitLatch.current = false;
      if (error instanceof ApiError && error.status === 409) {
        toast({
          title: "Your team is full",
          description:
            "You've reached the limit of active experts. Archive one from your team page to raise another.",
          variant: "destructive",
        });
        return;
      }
      toast({
        title: `Couldn't raise ${draft.name || "your expert"}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    step: draft.step,
    messages: buildTranscript(draft),
    name: draft.name,
    voiceLabel: draft.voiceLabel,
    firstJob: draft.firstJob,
    isSubmitting: isPending,
    submitName,
    pickVoice,
    skipVoice,
    pickFirstJob,
    skipFirstJob,
    goBack,
    finish,
  };
}
