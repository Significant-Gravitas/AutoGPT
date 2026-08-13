"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { BriefingCard } from "@/components/organisms/BriefingCard/BriefingCard";
import type { ReactNode } from "react";
import { useCopilotHome } from "./useCopilotHome";

interface Props {
  fallback: ReactNode;
}

// The short briefing recap above the composer. It is a recap only — the
// decisions inbox and team status live on /home — so the copilot's empty
// state keeps its onboarding surface (welcome dialog, greeting flow,
// suggestion themes) and the composer's recipient picker.
export function CopilotHome({ fallback }: Props) {
  const {
    briefing,
    isLoadingBriefing,
    isBriefingError,
    refetchBriefing,
    hasBriefing,
  } = useCopilotHome();

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
        className="mb-5"
      />
    );
  }

  if (!hasBriefing || !briefing) return <>{fallback}</>;

  return <BriefingCard briefing={briefing} className="mb-5" />;
}
