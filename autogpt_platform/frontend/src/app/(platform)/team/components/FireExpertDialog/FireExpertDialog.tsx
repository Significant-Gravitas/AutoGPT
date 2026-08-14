"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { getFireSummary } from "./helpers";
import {
  FireExpertFooter,
  FireExpertPreview,
} from "./FireExpertDialogSections";
import { useFireExpertDialog } from "./useFireExpertDialog";

interface Props {
  expertId: string;
  expertName: string;
  open: boolean;
  onClose: () => void;
  onFired?: () => void;
}

export function FireExpertDialog({
  expertId,
  expertName,
  open,
  onClose,
  onFired,
}: Props) {
  const {
    preview,
    isPreviewLoading,
    isPreviewError,
    isPreviewReady,
    retryPreview,
    isFiring,
    handleFire,
  } = useFireExpertDialog({
    expertId,
    expertName,
    open,
    onClose,
    onFired,
  });
  const summary = getFireSummary(preview);

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen) onClose();
  }

  return (
    <Dialog
      controlled={{ isOpen: open, set: handleOpenChange }}
      styling={{ maxWidth: "30rem" }}
      title={`Fire ${expertName}?`}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <FireExpertPreview
            expertName={expertName}
            automationLine={summary.automationLine}
            items={summary.items}
            isLoading={isPreviewLoading}
            isError={isPreviewError}
            isReady={isPreviewReady}
            onRetry={retryPreview}
          />
          <FireExpertFooter
            expertName={expertName}
            isFiring={isFiring}
            isPreviewLoading={isPreviewLoading}
            onClose={onClose}
            onFire={handleFire}
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
