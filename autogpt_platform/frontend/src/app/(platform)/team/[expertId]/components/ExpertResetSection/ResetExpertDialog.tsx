"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ReactNode } from "react";
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
  const { isResetting, handleReset } = useResetExpertDialog({
    expertId,
    expertName,
    onClose,
  });

  // While the reset runs, ESC and overlay clicks must not dismiss the
  // dialog, or the user loses the result of an action they cannot undo.
  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen && !isResetting) onClose();
  }

  return (
    <Dialog
      variant="compact"
      controlled={{ isOpen: open, set: handleOpenChange }}
      styling={{ maxWidth: "30rem" }}
      title={`Reset ${expertName}?`}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" tone="secondary">
            {expertName} goes back to how they were the day you hired them. This
            can&apos;t be undone. If {expertName} is working on something, wait
            for them to finish first.
          </Text>
          <ResetGroup label="Erased">
            <ResetLine>Everything {expertName} remembers</ResetLine>
            <ResetLine>
              {expertName}&apos;s computer and the notes on it
            </ResetLine>
            {hasTemplate ? (
              <ResetLine>Your edits to their soul and routines</ResetLine>
            ) : null}
          </ResetGroup>
          <ResetGroup label="Archived">
            <ResetLine>
              Your chats with {expertName} leave your chat list
            </ResetLine>
          </ResetGroup>
          <ResetGroup label="Kept">
            <ResetLine>
              Their workflows, connections, budget and run history
            </ResetLine>
            <ResetLine>
              Your files, and what the rest of your team knows
            </ResetLine>
            {hasTemplate ? null : (
              <ResetLine>
                Their soul and routines, since you created {expertName}
              </ResetLine>
            )}
          </ResetGroup>
          <Dialog.Footer>
            <Button
              size="small"
              variant="secondary"
              disabled={isResetting}
              onClick={onClose}
            >
              Cancel
            </Button>
            <Button
              size="small"
              variant="destructive"
              loading={isResetting}
              onClick={handleReset}
              data-testid="reset-expert-confirm"
            >
              Reset {expertName}
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

interface GroupProps {
  label: string;
  children: ReactNode;
}

function ResetGroup({ label, children }: GroupProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <Text variant="small-medium" tone="muted">
        {label}
      </Text>
      <ul className="flex flex-col gap-1.5">{children}</ul>
    </div>
  );
}

function ResetLine({ children }: { children: ReactNode }) {
  return (
    <li className="flex items-start gap-2.5">
      <span
        aria-hidden
        className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-border-strong"
      />
      <Text variant="body" tone="secondary">
        {children}
      </Text>
    </li>
  );
}
