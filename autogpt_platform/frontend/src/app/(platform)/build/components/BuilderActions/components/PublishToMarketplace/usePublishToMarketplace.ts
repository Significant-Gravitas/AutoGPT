import { useState, useCallback } from "react";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";

export type PublishStep = "select" | "info" | "review";

export type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmission | null;
};

const defaultPublishState: PublishState = {
  isOpen: false,
  step: "select",
  submissionData: null,
};

interface UsePublishToMarketplaceProps {
  flowID: string | null;
  flowVersion: number | null;
}

export function usePublishToMarketplace({
  flowID,
  flowVersion,
}: UsePublishToMarketplaceProps) {
  const [publishState, setPublishState] =
    useState<PublishState>(defaultPublishState);

  function handlePublishToMarketplace() {
    if (!flowID || flowVersion === null) return;

    // Builder already knows the agent — skip the picker and jump straight
    // to the listing form pre-scoped to the current graph/version.
    setPublishState({
      isOpen: true,
      step: "info",
      submissionData: null,
    });
  }

  const handleStateChange = useCallback((newState: PublishState) => {
    setPublishState(newState);
  }, []);

  return {
    handlePublishToMarketplace,
    publishState,
    handleStateChange,
  };
}
