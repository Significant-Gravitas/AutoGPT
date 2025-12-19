import { useState, useCallback } from "react";

export type PublishStep = "select" | "info" | "review";

export type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: any | null;
};

const defaultPublishState: PublishState = {
  isOpen: false,
  step: "select",
  submissionData: null,
};

interface UsePublishToMarketplaceProps {
  flowID: string | null;
}

export function usePublishToMarketplace({
  flowID,
}: UsePublishToMarketplaceProps) {
  const [publishState, setPublishState] =
    useState<PublishState>(defaultPublishState);

  const handlePublishToMarketplace = () => {
    if (!flowID) return;

    // Open the publish modal starting with the select step
    setPublishState({
      isOpen: true,
      step: "select",
      submissionData: null,
    });
  };

  const handleStateChange = useCallback((newState: PublishState) => {
    setPublishState(newState);
  }, []);

  return {
    handlePublishToMarketplace,
    publishState,
    handleStateChange,
  };
}
