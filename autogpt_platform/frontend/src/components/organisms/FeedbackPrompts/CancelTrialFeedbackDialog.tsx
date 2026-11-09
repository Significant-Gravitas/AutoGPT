"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ReactNode } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { OttoOfferCard } from "./components/OttoOfferCard";
import { ReasonTiles } from "./components/ReasonTiles";
import { TRIAL_CANCEL_REASONS } from "./helpers";
import { type Step, useCancelTrialFeedback } from "./useCancelTrialFeedback";

const TITLES: Record<Step, string> = {
  reason: "Before you go, what's the main reason?",
  offer: "Let Otto show you first",
  confirm: "Cancel your trial?",
};

interface Props {
  trigger?: ReactNode;
  endDate?: string;
  isCanceling?: boolean;
  starters?: string[];
  initialStep?: Step;
  initialReason?: string | null;
  onOpen?: () => void;
  onReason?: (value: string) => void;
  onNote?: (note: string) => void;
  onConfirm?: () => void;
  onPickStarter?: (starter: string) => void;
  onTryOtto?: () => void;
}

export function CancelTrialFeedbackDialog({
  trigger,
  endDate = "November 4",
  isCanceling = false,
  starters = [],
  initialStep = "reason",
  initialReason = null,
  onOpen,
  onReason,
  onNote,
  onConfirm,
  onPickStarter,
  onTryOtto,
}: Props) {
  const flow = useCancelTrialFeedback({
    startsOpen: !trigger,
    initialStep,
    initialReason,
    onOpen,
    onReason,
    onNote,
    onConfirm,
    onPickStarter,
    onTryOtto,
  });

  return (
    <Dialog
      title={TITLES[flow.step]}
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{ isOpen: flow.isOpen, set: flow.setOpen }}
    >
      {trigger ? <Dialog.Trigger>{trigger}</Dialog.Trigger> : null}
      <Dialog.Content>
        <NoInitialFocus />
        {flow.step === "reason" ? (
          <div className="flex flex-col gap-4">
            <Text variant="body" tone="muted">
              One tap helps us fix the right thing.
            </Text>
            <ReasonTiles
              reasons={TRIAL_CANCEL_REASONS}
              label={TITLES.reason}
              selected={flow.reason}
              onSelect={flow.pickReason}
            />
            {flow.asksForNote ? (
              <Textarea
                rows={3}
                value={flow.note}
                onChange={(e) => flow.setNote(e.target.value)}
                aria-label="What would have made you stay?"
                placeholder="What would have made you stay?"
                className="resize-none"
              />
            ) : null}
          </div>
        ) : null}

        {flow.step === "offer" ? (
          <OttoOfferCard starters={starters} onPick={flow.pickStarter} />
        ) : null}

        {flow.step === "confirm" ? (
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
        ) : null}

        <Dialog.Footer className="items-center gap-2">
          {flow.step === "reason" ? (
            <>
              <Button
                variant="ghost"
                size="small"
                className="mr-auto"
                onClick={() => flow.setStep("confirm")}
              >
                Skip
              </Button>
              <Button
                variant="outline"
                size="small"
                onClick={() => flow.setOpen(false)}
              >
                Keep trial
              </Button>
              <Button
                size="small"
                disabled={!flow.reason}
                onClick={flow.continueFromReason}
              >
                Continue
              </Button>
            </>
          ) : null}
          {flow.step === "offer" ? (
            <>
              <Button
                variant="ghost"
                size="small"
                onClick={() => flow.setStep("confirm")}
              >
                No thanks, cancel
              </Button>
              <Button size="small" onClick={flow.tryOtto}>
                Try it with Otto
              </Button>
            </>
          ) : null}
          {flow.step === "confirm" ? (
            <>
              <Button
                variant="outline"
                size="small"
                onClick={() => flow.setOpen(false)}
              >
                Keep trial
              </Button>
              <Button
                size="small"
                disabled={isCanceling}
                onClick={flow.confirm}
              >
                Cancel trial
              </Button>
            </>
          ) : null}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
