import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { useState } from "react";
import {
  beatTriggers,
  buildFlowItems,
  clearedAnswer,
  lastAnsweredBeat,
  type BeatKey,
} from "./flowItems";
import {
  assembledKit,
  clearDraft,
  EMPTY_DRAFT,
  loadDraft,
  resolveVoicePreferences,
  saveDraft,
  voiceSummaryLabel,
  VOICE_SAMPLES,
  VOICE_SKIPPED_LABEL,
  type RaiseAttachmentDraft,
  type RaiseDraft,
  type RaiseKit,
} from "./helpers";
import { useFlowProgress } from "./useFlowProgress";
import { useRaiseSubmission } from "./useRaiseSubmission";

export function useRaisePage() {
  const [draft, setDraft] = useState<RaiseDraft>(loadDraft);
  const progress = useFlowProgress(beatTriggers(draft));
  const { finish: submitRaise, isSubmitting } = useRaiseSubmission();

  function finish(kit: RaiseKit) {
    void submitRaise(draft, kit);
  }

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
      step: "budget",
    });
  }

  function skipVoice() {
    update({
      voicePreferences: "",
      voiceLabel: VOICE_SKIPPED_LABEL,
      step: "budget",
    });
  }

  function submitBudget(credits: number) {
    update({ budget: { credits }, step: "marketplace" });
  }

  function skipBudget() {
    update({ budget: { credits: null }, step: "marketplace" });
  }

  function submitMarketplace(attachments: RaiseAttachmentDraft[]) {
    update({ marketplace: attachments, step: "skills" });
  }

  function skipMarketplace() {
    update({ marketplace: [], step: "skills" });
  }

  // Skills stays unanswered in the draft until the raise succeeds, and a
  // successful raise clears the draft. Recording it up front would render the
  // step as answered and take away the retry control when the POST fails.
  function submitSkills(attachments: RaiseAttachmentDraft[]) {
    finish({
      weeklyBudget: draft.budget?.credits ?? null,
      attachments: [...(draft.marketplace ?? []), ...attachments],
    });
  }

  function skipSkills() {
    finish({
      weeklyBudget: draft.budget?.credits ?? null,
      attachments: draft.marketplace ?? [],
    });
  }

  function goBack() {
    const beat = lastAnsweredBeat(draft);
    if (!beat) return;
    update({ ...clearedAnswer(beat), step: beat });
    progress.clearAfter(beat);
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
    budget: draft.budget,
    marketplace: draft.marketplace,
    skills: draft.skills,
    kit: assembledKit(draft),
    isSubmitting,
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
    submitBudget,
    skipBudget,
    submitMarketplace,
    skipMarketplace,
    submitSkills,
    skipSkills,
    goBack,
  };
}
