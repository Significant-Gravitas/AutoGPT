"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { NeedsAttentionList } from "@/components/organisms/NeedsAttentionList/NeedsAttentionList";
import type { ReactNode } from "react";
import { BriefingCard } from "./components/BriefingCard/BriefingCard";
import { TeamStrip } from "./components/TeamStrip/TeamStrip";
import { useCopilotHome } from "./useCopilotHome";

interface Props {
  fallback: ReactNode;
}

// The briefing-first block of the copilot home. Mounted *inside*
// EmptySession rather than beside it, so the experts cohort keeps the
// onboarding surface (welcome dialog, greeting flow, suggestion themes) and
// the composer's recipient picker, which live there.
export function CopilotHome({ fallback }: Props) {
  const {
    briefing,
    isLoadingBriefing,
    isBriefingError,
    refetchBriefing,
    hasBriefing,
  } = useCopilotHome();

  return (
    <>
      {/* Briefing card slot — falls back to whatever the caller renders in
          its place only when a successful fetch says there is no briefing
          yet; a failed fetch shows the error card so it can't masquerade as
          "no briefing". Nothing renders until load settles so the fallback
          doesn't flash for users who do have a briefing. */}
      {isLoadingBriefing ? null : isBriefingError ? (
        <ErrorCard
          context="briefing"
          httpError={{ message: "Failed to load your briefing" }}
          onRetry={() => refetchBriefing()}
          className="mb-5"
        />
      ) : hasBriefing && briefing ? (
        <BriefingCard briefing={briefing} />
      ) : (
        fallback
      )}

      <NeedsAttentionList />

      <TeamStrip />
    </>
  );
}
