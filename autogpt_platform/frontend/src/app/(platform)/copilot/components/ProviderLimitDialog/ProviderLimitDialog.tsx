"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import type { ProviderFailure } from "../../providerFailure";
import { useProviderLimitDialog } from "./useProviderLimitDialog";

interface Props {
  failure: ProviderFailure | null;
  sessionId: string | null;
  onDismiss: () => void;
}

/**
 * A linked subscription stopped accepting turns. Offer to keep going.
 *
 * The PRD's out-of-limit path: the run pauses and offers to continue
 * somewhere else, and never moves the bill on its own. A dialog rather than a
 * toast because the choice has a cost — the rest of this chat spends AutoGPT
 * credits instead of a plan the user already pays for — and that is not a
 * thing to slip past someone in the corner of the screen.
 *
 * Waiting stays a real option. When the provider told us when the limit
 * resets that is said here; an invented time would be worse than none,
 * because people plan around it.
 */
export function ProviderLimitDialog({ failure, sessionId, onDismiss }: Props) {
  const {
    alternative,
    continueHere,
    isSwitching,
    resetHint,
    isLoadingOffers,
    failedToLoadOffers,
    retryOffers,
  } = useProviderLimitDialog({ failure, sessionId, onDismiss });

  return (
    <Dialog
      title="You've hit this connection's limit"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen: Boolean(failure),
        set: (open) => {
          if (!open) onDismiss();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body" className="text-zinc-600">
          {failure?.message?.trim() ||
            "The connection this chat runs on stopped accepting turns."}
          {resetHint ? ` ${resetHint}` : ""}
        </Text>

        {isLoadingOffers ? (
          <Text variant="small" className="mt-3 !text-zinc-500">
            Checking what else this chat can run on&hellip;
          </Text>
        ) : failedToLoadOffers ? (
          <>
            <Text variant="small" className="mt-3 !text-zinc-500">
              We couldn&apos;t check your other connections just now.
            </Text>
            <Dialog.Footer>
              <Button variant="secondary" onClick={onDismiss}>
                Close
              </Button>
              <Button variant="primary" onClick={retryOffers}>
                Try again
              </Button>
            </Dialog.Footer>
          </>
        ) : alternative ? (
          <>
            <Text variant="small" className="mt-3 !text-zinc-500">
              Continuing moves the rest of this chat to{" "}
              {alternative.display_name}. Everything already said stays as it
              is, on the connection it ran on.
            </Text>
            <Dialog.Footer>
              <Button variant="secondary" onClick={onDismiss}>
                Wait for it to reset
              </Button>
              <Button
                variant="primary"
                onClick={continueHere}
                loading={isSwitching}
              >
                Continue on {alternative.display_name}
              </Button>
            </Dialog.Footer>
          </>
        ) : (
          <>
            {/* Nothing to offer: saying so is better than a button that
                cannot do anything. */}
            <Text variant="small" className="mt-3 !text-zinc-500">
              There is no other connection set up to continue on. You can add
              one in Settings, or wait for this one to reset.
            </Text>
            <Dialog.Footer>
              <Button variant="primary" onClick={onDismiss}>
                Close
              </Button>
            </Dialog.Footer>
          </>
        )}
      </Dialog.Content>
    </Dialog>
  );
}
