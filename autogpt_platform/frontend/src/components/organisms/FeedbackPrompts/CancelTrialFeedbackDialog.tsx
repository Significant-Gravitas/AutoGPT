"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { OttoOfferCard } from "./components/OttoOfferCard";
import { ReasonTiles } from "./components/ReasonTiles";
import { OTHER_LABEL, TRIAL_CANCEL_REASONS } from "./helpers";

type Step = "reason" | "offer" | "confirm";

const OFFER_REASON = "I haven't seen results yet";

const TITLES: Record<Step, string> = {
  reason: "Before you go, what's the main reason?",
  offer: "Let Otto show you first",
  confirm: "Cancel your trial?",
};

interface Props {
  starters?: string[];
  initialStep?: Step;
  initialReason?: string | null;
}

export function CancelTrialFeedbackDialog({
  starters = [],
  initialStep = "reason",
  initialReason = null,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [step, setStep] = useState<Step>(initialStep);
  const [reason, setReason] = useState<string | null>(initialReason);
  const [note, setNote] = useState("");

  function close() {
    setIsOpen(false);
  }

  return (
    <Dialog
      title={TITLES[step]}
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Content>
        <NoInitialFocus />
        {step === "reason" ? (
          <div className="flex flex-col gap-4">
            <Text variant="body" className="!text-foreground-tertiary">
              One tap helps us fix the right thing.
            </Text>
            <ReasonTiles
              reasons={TRIAL_CANCEL_REASONS}
              label={TITLES.reason}
              selected={reason}
              onSelect={setReason}
            />
            {reason === OTHER_LABEL || reason === "It's missing something" ? (
              <Textarea
                rows={3}
                value={note}
                onChange={(e) => setNote(e.target.value)}
                aria-label="What would have made you stay?"
                placeholder="What would have made you stay?"
                className="resize-none"
              />
            ) : null}
          </div>
        ) : null}

        {step === "offer" ? <OttoOfferCard starters={starters} /> : null}

        {step === "confirm" ? (
          <div className="flex flex-col gap-3">
            <Text variant="body" className="!text-foreground-emphasis">
              Your trial won&apos;t convert to a paid plan and your card
              won&apos;t be charged.
            </Text>
            <Text variant="body" className="!text-foreground-emphasis">
              You keep full access until{" "}
              <strong className="font-semibold">November 4</strong>, and you can
              resume your trial any time before then.
            </Text>
          </div>
        ) : null}

        <Dialog.Footer className="items-center gap-2">
          {step === "reason" ? (
            <>
              <Button
                variant="ghost"
                size="small"
                className="mr-auto"
                onClick={() => setStep("confirm")}
              >
                Skip
              </Button>
              <Button variant="outline" size="small" onClick={close}>
                Keep trial
              </Button>
              <Button
                size="small"
                disabled={!reason}
                onClick={() =>
                  setStep(reason === OFFER_REASON ? "offer" : "confirm")
                }
              >
                Continue
              </Button>
            </>
          ) : null}
          {step === "offer" ? (
            <>
              <Button
                variant="ghost"
                size="small"
                onClick={() => setStep("confirm")}
              >
                No thanks, cancel
              </Button>
              <Button size="small" onClick={close}>
                Try it with Otto
              </Button>
            </>
          ) : null}
          {step === "confirm" ? (
            <>
              <Button variant="outline" size="small" onClick={close}>
                Keep trial
              </Button>
              <Button size="small" onClick={close}>
                Cancel trial
              </Button>
            </>
          ) : null}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
