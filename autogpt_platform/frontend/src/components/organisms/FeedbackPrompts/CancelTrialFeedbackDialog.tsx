"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ReactNode } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { ReasonTiles } from "./components/ReasonTiles";
import { MAX_FEEDBACK_NOTE_CHARS, TRIAL_CANCEL_REASONS } from "./helpers";
import {
  type CancelTrialAnswer,
  useCancelTrialFeedback,
} from "./useCancelTrialFeedback";

const REASON_LABEL = "What's the main reason? (optional)";

interface Props {
  trigger?: ReactNode;
  endDate?: string;
  isCanceling?: boolean;
  initialReason?: string | null;
  onOpen?: () => void;
  onReason?: (value: string) => void;
  onConfirm?: (answer: CancelTrialAnswer) => void;
}

// One dialog, with Keep trial and Cancel trial always in reach: the reason is
// optional and never stands between someone and the cancel button.
export function CancelTrialFeedbackDialog({
  trigger,
  endDate = "November 4",
  isCanceling = false,
  initialReason = null,
  onOpen,
  onReason,
  onConfirm,
}: Props) {
  const flow = useCancelTrialFeedback({
    startsOpen: !trigger,
    initialReason,
    onOpen,
    onReason,
    onConfirm,
  });

  return (
    <Dialog
      title="Cancel your trial?"
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{ isOpen: flow.isOpen, set: flow.setOpen }}
    >
      {trigger ? <Dialog.Trigger>{trigger}</Dialog.Trigger> : null}
      <Dialog.Content>
        <NoInitialFocus />
        <div className="flex flex-col gap-5">
          <div
            data-slot="cancel-trial-dialog-body"
            className="flex flex-col gap-3"
          >
            <Text variant="body" tone="emphasis">
              Your trial won&apos;t convert to a paid plan and your card
              won&apos;t be charged.
            </Text>
            <Text variant="body" tone="emphasis">
              You keep full access until{" "}
              <strong
                data-slot="cancel-trial-dialog-end-date"
                className="font-semibold"
              >
                {endDate}
              </strong>
              , and you can resume your trial any time before then.
            </Text>
          </div>
          <div className="flex flex-col gap-3">
            <Text variant="body-medium" tone="primary">
              {REASON_LABEL}
            </Text>
            <ReasonTiles
              reasons={TRIAL_CANCEL_REASONS}
              label={REASON_LABEL}
              selected={flow.reason}
              onSelect={flow.pickReason}
              size="compact"
            />
            {flow.asksForNote ? (
              <Textarea
                maxLength={MAX_FEEDBACK_NOTE_CHARS}
                rows={2}
                value={flow.note}
                onChange={(e) => flow.setNote(e.target.value)}
                aria-label="Tell us a bit more (optional)"
                placeholder="Tell us a bit more (optional)"
                className="min-h-0 resize-none"
              />
            ) : null}
          </div>
        </div>
        <Dialog.Footer className="items-center gap-2">
          <Button
            variant="outline"
            size="small"
            onClick={() => flow.setOpen(false)}
          >
            Keep trial
          </Button>
          <Button size="small" disabled={isCanceling} onClick={flow.confirm}>
            Cancel trial
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
