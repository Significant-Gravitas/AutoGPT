import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { useSearchParams } from "next/navigation";
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
  draftWithPrefilledRole,
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
import { useSkillsAvailability } from "./useSkillsAvailability";

export function useRaisePage() {
  const searchParams = useSearchParams();
  // Seeded in the initialiser rather than an effect: an effect would render
  // the role question first and then snatch it away.
  const [draft, setDraft] = useState<RaiseDraft>(() =>
    draftWithPrefilledRole(loadDraft(), searchParams.get("role")),
  );
  const { hasSkillsToOffer } = useSkillsAvailability();
  const hasSkillsBeat = draft.marketplace !== null || hasSkillsToOffer;
  const progress = useFlowProgress(beatTriggers(draft, hasSkillsBeat));
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
    update({ role: roleId, step: "jobTitle" });
  }

  function submitJobTitle(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    update({ jobTitle: trimmed, step: "name" });
  }

  function skipJobTitle() {
    update({ jobTitle: "", step: "name" });
  }

  function submitName(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    update({ name: trimmed, step: "avatar" });
  }

  function pickAvatar(avatarUrl: string, colorId: string) {
    update({ avatarUrl, color: colorId, step: "about" });
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

  // With no skills to offer the marketplace beat is the last one, so it
  // raises the expert itself rather than handing off to a beat that never
  // opens. As with skills, the answer is not recorded until the POST wins,
  // which keeps the retry control on screen when it fails.
  function submitMarketplace(attachments: RaiseAttachmentDraft[]) {
    if (!hasSkillsToOffer) {
      finish({ weeklyBudget: draft.budget?.credits ?? null, attachments });
      return;
    }
    update({ marketplace: attachments, step: "skills" });
  }

  function skipMarketplace() {
    if (!hasSkillsToOffer) {
      finish({ weeklyBudget: draft.budget?.credits ?? null, attachments: [] });
      return;
    }
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
    jobTitle: draft.jobTitle,
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
    isMarketplaceFinal: !hasSkillsBeat,
    isSubmitting,
    canGoBack: lastAnsweredBeat(draft) !== null,
    startRaising,
    restart,
    revealStep: (beat: BeatKey) => progress.revealStep(beat),
    pickRole,
    submitJobTitle,
    skipJobTitle,
    submitName,
    pickAvatar,
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
