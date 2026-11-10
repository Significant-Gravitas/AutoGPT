import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ReactNode } from "react";

interface Props {
  expertName: string;
  hasTemplate: boolean;
  isResetting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ResetConfirm({
  expertName,
  hasTemplate,
  isResetting,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" tone="secondary">
        {hasTemplate
          ? `${expertName} goes back to how they were the day you hired them.`
          : `${expertName} goes back to how they were the day you created them, keeping their soul and routines.`}{" "}
        This can&apos;t be undone. If {expertName} is working on something, wait
        for them to finish first.
      </Text>
      <ResetGroup label="Erased">
        <ResetLine>Everything {expertName} remembers</ResetLine>
        <ResetLine>{expertName}&apos;s computer and the notes on it</ResetLine>
        {hasTemplate ? (
          <>
            <ResetLine>Your edits to their soul and routines</ResetLine>
            <ResetLine>
              Routines you switched on are switched off, and the ones{" "}
              {expertName} came with come back, switched off
            </ResetLine>
          </>
        ) : null}
      </ResetGroup>
      <ResetGroup label="Removed">
        <ResetLine>
          Your chats with {expertName}. They can&apos;t be opened again, even
          from an old link
        </ResetLine>
        <ResetLine>Approvals from those chats waiting in Needs you</ResetLine>
      </ResetGroup>
      <ResetGroup label="Kept">
        <ResetLine>The skills {expertName} has learned</ResetLine>
        <ResetLine>Their workflows, and the schedules on them</ResetLine>
        <ResetLine>Their connections, budget and run history</ResetLine>
        <ResetLine>Your files, and what the rest of your team knows</ResetLine>
        {hasTemplate ? null : (
          <ResetLine>
            Their soul and routines, each routine still on or off as it is now
          </ResetLine>
        )}
      </ResetGroup>
      <Dialog.Footer>
        <Button
          size="small"
          variant="secondary"
          disabled={isResetting}
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button
          size="small"
          variant="destructive"
          loading={isResetting}
          onClick={onConfirm}
          data-testid="reset-expert-confirm"
        >
          Reset {expertName}
        </Button>
      </Dialog.Footer>
    </div>
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
