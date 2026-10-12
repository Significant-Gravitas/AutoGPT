"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ConfirmDeleteStep } from "./components/ConfirmDeleteStep/ConfirmDeleteStep";
import { ExitReasonStep } from "./components/ExitReasonStep/ExitReasonStep";
import type { DeletionSource } from "./helpers";
import { useDeleteAccountFlow } from "./useDeleteAccountFlow";

interface Props {
  source: DeletionSource;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DeleteAccountFlow({ source, open, onOpenChange }: Props) {
  const {
    step,
    session,
    title,
    isDeleting,
    setIsDeleting,
    handleOpenChange,
    close,
    goToConfirm,
  } = useDeleteAccountFlow({ open, onOpenChange });

  return (
    <Dialog
      title={title}
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      // Autocapture would tie the anonymous exit answer to the person. The
      // scroll padding keeps a row or button reached by keyboard from landing
      // under the sticky footer on short screens.
      className="ph-no-capture [&_.overflow-y-auto]:scroll-pb-24"
      autoFocus
      forceOpen={isDeleting}
      controlled={{ isOpen: open, set: handleOpenChange }}
    >
      <Dialog.Content>
        {step === "reason" ? (
          <ExitReasonStep
            key={session}
            origin={source}
            onClose={close}
            onContinue={goToConfirm}
          />
        ) : (
          <ConfirmDeleteStep
            key={session}
            source={source}
            onClose={close}
            onBusyChange={setIsDeleting}
          />
        )}
      </Dialog.Content>
    </Dialog>
  );
}
