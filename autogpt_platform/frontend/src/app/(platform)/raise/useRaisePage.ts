import { useCreateRaisedExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import {
  beatTriggers,
  buildFlowItems,
  clearedAnswer,
  lastAnsweredBeat,
  type BeatKey,
} from "./flowItems";
import {
  clearDraft,
  EMPTY_DRAFT,
  getExpertLimitCode,
  loadDraft,
  resolveVoicePreferences,
  saveDraft,
  voiceSummaryLabel,
  VOICE_SAMPLES,
  VOICE_SKIPPED_LABEL,
  type RaiseDraft,
} from "./helpers";
import { useFlowProgress } from "./useFlowProgress";

export function useRaisePage() {
  const router = useRouter();
  const { mutateAsync: createRaisedExpert, isPending } =
    useCreateRaisedExpert();
  const [draft, setDraft] = useState<RaiseDraft>(loadDraft);
  const progress = useFlowProgress(beatTriggers(draft));
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

  function startRaising() {
    update({ hasStarted: true });
  }

  function restart() {
    clearDraft();
    setDraft(EMPTY_DRAFT);
    progress.reset();
  }

  function pickRole(roleId: string) {
    update({ role: roleId, step: "name" });
  }

  function submitName(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    update({ name: trimmed, step: "color" });
  }

  function pickColor(colorId: string) {
    update({ color: colorId, step: "avatar" });
  }

  function pickAvatar(avatarUrl: string) {
    update({ avatarUrl, step: "about" });
  }

  function skipAvatar() {
    update({ avatarUrl: "", step: "about" });
  }

  function submitAbout(value: string) {
    update({ about: value.trim(), step: "voice" });
  }

  function skipAbout() {
    update({ about: "", step: "voice" });
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
      step: "firstTask",
    });
  }

  function skipVoice() {
    update({
      voicePreferences: "",
      voiceLabel: VOICE_SKIPPED_LABEL,
      step: "firstTask",
    });
  }

  // The first task is the payoff, so submitting it finishes the raise and
  // carries straight into the chat instead of parking on a review screen.
  function submitFirstTask(task: string) {
    update({ firstTask: task, step: "done" });
    finish(task);
  }

  function skipFirstTask() {
    update({ firstTask: "", step: "done" });
    finish("");
  }

  function goBack() {
    const beat = lastAnsweredBeat(draft);
    if (!beat) return;
    update({ ...clearedAnswer(beat), step: beat });
    progress.clearAfter(beat);
  }

  async function finish(firstTask: string) {
    if (submitLatch.current) return;
    submitLatch.current = true;
    setIsSubmissionLocked(true);
    try {
      const response = await createRaisedExpert({
        data: {
          name: draft.name,
          role: draft.role,
          color: draft.color,
          avatar_url: draft.avatarUrl || null,
          about: draft.about || null,
          voice_preferences: draft.voicePreferences || null,
        },
      });
      const result = response.data as RaiseResult;
      clearDraft();
      router.push(chatHandoffUrl(result.expert.id, firstTask));
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
    hasStarted: draft.hasStarted,
    role: draft.role,
    color: draft.color,
    avatarUrl: draft.avatarUrl,
    about: draft.about,
    voiceLabel: draft.voiceLabel,
    items: buildFlowItems(draft, progress),
    name: draft.name,
    firstTask: draft.firstTask,
    isSubmitting: isPending || isSubmissionLocked,
    canGoBack: lastAnsweredBeat(draft) !== null,
    startRaising,
    restart,
    revealStep: (beat: BeatKey) => progress.revealStep(beat),
    pickRole,
    submitName,
    pickColor,
    pickAvatar,
    skipAvatar,
    submitAbout,
    skipAbout,
    pickVoice,
    skipVoice,
    submitFirstTask,
    skipFirstTask,
    goBack,
  };
}

// CoPilot picks up `#prompt=` on mount and sends it as the first message when
// `autosubmit=true`, so the raise flow hands the task over without needing a
// session to exist yet.
function chatHandoffUrl(expertId: string, firstTask: string) {
  const base = `/copilot?expertId=${encodeURIComponent(expertId)}`;
  if (!firstTask) return base;
  return `${base}&autosubmit=true#prompt=${encodeURIComponent(firstTask)}`;
}
