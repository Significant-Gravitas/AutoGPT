"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { BriefingCard } from "@/components/organisms/BriefingCard/BriefingCard";
import { hasRecapContent } from "@/components/organisms/BriefingCard/helpers";
import { useCopilotHome } from "./useCopilotHome";

// The short briefing recap under the composer. It is a recap only — the
// decisions inbox and team status live on /home — so the copilot's empty
// state keeps its onboarding surface (welcome dialog, greeting flow,
// suggestion themes) and the composer's recipient picker.
export function CopilotHome() {
  const { briefing, isLoadingBriefing, isBriefingError, refetchBriefing } =
    useCopilotHome();

  // Nothing renders until load settles so the fallback doesn't flash for
  // users who do have a briefing. A failed fetch shows the error card so it
  // can't masquerade as "no briefing".
  if (isLoadingBriefing) return null;

  if (isBriefingError) {
    return (
      <ErrorCard
        context="briefing"
        httpError={{ message: "Failed to load your briefing" }}
        onRetry={() => refetchBriefing()}
      />
    );
  }

  // Not just "no briefing": a briefing of nothing but pending decisions has
  // no runs to recap, and the card would render null. The inbox that used to
  // carry that case lives on /home now, as does the workflow-runs strip that
  // once stood in here, so the empty state simply stays quiet.
  if (!hasRecapContent(briefing)) return null;

  return <BriefingCard briefing={briefing} />;
}
