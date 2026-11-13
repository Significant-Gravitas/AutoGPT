"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Clock01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { ResetBusy } from "./components/ResetBusy";
import { ResetConfirm } from "./components/ResetConfirm";
import { ResetDone } from "./components/ResetDone";
import { StatusTitle } from "./components/StatusTitle";
import { useResetExpertDialog } from "./useResetExpertDialog";

interface Props {
  expertId: string;
  expertName: string;
  hasTemplate: boolean;
  open: boolean;
  onClose: () => void;
}

export function ResetExpertDialog({
  expertId,
  expertName,
  hasTemplate,
  open,
  onClose,
}: Props) {
  const { phase, isResetting, handleReset } = useResetExpertDialog({
    expertId,
    expertName,
    open,
  });

  // While the reset runs, ESC and overlay clicks must not dismiss the
  // dialog, or the user loses the result of an action they cannot undo.
  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen && !isResetting) onClose();
  }

  const title =
    phase === "done" ? (
      <StatusTitle tone="success" icon={Tick02Icon}>
        {expertName} is reset
      </StatusTitle>
    ) : phase === "busy" ? (
      <StatusTitle tone="warning" icon={Clock01Icon}>
        {expertName} is still working
      </StatusTitle>
    ) : (
      `Reset ${expertName}?`
    );

  return (
    <Dialog
      variant="compact"
      controlled={{ isOpen: open, set: handleOpenChange }}
      styling={{ maxWidth: "30rem" }}
      title={title}
    >
      <Dialog.Content>
        {phase === "done" ? (
          <ResetDone
            expertId={expertId}
            expertName={expertName}
            onDone={onClose}
          />
        ) : phase === "busy" ? (
          <ResetBusy
            expertName={expertName}
            isRetrying={isResetting}
            onClose={onClose}
            onRetry={handleReset}
          />
        ) : (
          <ResetConfirm
            expertName={expertName}
            hasTemplate={hasTemplate}
            isResetting={isResetting}
            onCancel={onClose}
            onConfirm={handleReset}
          />
        )}
      </Dialog.Content>
    </Dialog>
  );
}
