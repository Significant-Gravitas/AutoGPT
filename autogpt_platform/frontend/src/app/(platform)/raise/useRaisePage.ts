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
  getExpertLimitCode,
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
  const [isSubmissionLocked, setIsSubmissionLocked] = useState(false);
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
      step: "firstJob",
    });
  }

  function skipVoice() {
    update({
      voicePreferences: "",
      voiceLabel: null,
      step: "firstJob",
    });
  }

  function pickFirstJob(job: { id: string; name: string }) {
    update({ firstJob: job, step: "review" });
  }

  function skipFirstJob() {
    update({ firstJob: null, step: "review" });
  }

  function goBack() {
    update({ step: previousStep(draft.step) });
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
      const result = response.data as RaiseResult;
      clearDraft();
      if (draft.firstJob && !result.first_job_installed) {
        if (result.first_job_failure_reason === "unavailable") {
          toast({
            title: `${draft.firstJob.name} is no longer available`,
            description: `${result.expert.name} is ready. You can choose another first job from their page.`,
            variant: "default",
          });
        } else {
          toast({
            title: `Couldn't set up ${result.expert.name}'s first job`,
            description: `You can install "${draft.firstJob.name}" from their page anytime.`,
            variant: "default",
          });
        }
      }
      const kickoff = result.expert.workflows.length > 0 ? "&kickoff=1" : "";
      router.push(`/copilot?expertId=${result.expert.id}${kickoff}`);
    } catch (error) {
      submitLatch.current = false;
      setIsSubmissionLocked(false);
      if (error instanceof ApiError && error.status === 409) {
        if (
          getExpertLimitCode(error.response) === "raised_expert_lifetime_limit"
        ) {
          toast({
            title: "Expert creation limit reached",
            description:
              "This account has reached its lifetime raised-expert limit. Contact support if you need more capacity.",
            variant: "destructive",
          });
          return;
        }
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
