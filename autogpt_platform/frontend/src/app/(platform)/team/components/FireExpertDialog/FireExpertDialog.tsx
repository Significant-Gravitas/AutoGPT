"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { PropsWithChildren } from "react";
import { getFireSummary } from "./helpers";
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

  function getAutomationLineText() {
    if (isPreviewLoading) return "Checking what will pause…";
    if (isPreviewError)
      return "We couldn't preview what pauses, but you can still let them go.";
    return summary.automationLine;
  }

  return (
    <Dialog
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
      styling={{ maxWidth: "30rem" }}
      title={`Fire ${expertName}?`}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-600">
            Here is exactly what happens when you let {expertName} go.
          </Text>
          <ul className="flex flex-col gap-2.5">
            <FireLine>Installed workflows stay in your library.</FireLine>
            <FireLine>{getAutomationLineText()}</FireLine>
            <FireLine>Chat threads become read-only history.</FireLine>
            <FireLine>Their work stays yours.</FireLine>
          </ul>
          {isPreviewError ? (
            <div className="flex justify-end">
              <Button
                variant="secondary"
                size="small"
                onClick={() => retryPreview()}
                data-testid="fire-preview-retry"
              >
                Retry preview
              </Button>
            </div>
          ) : null}
          {isPreviewReady && summary.items.length > 0 ? (
            <div className="rounded-xl bg-zinc-50 px-3.5 py-3 ring-1 ring-inset ring-zinc-200/80">
              <Text variant="small" className="text-zinc-500">
                Pausing now
              </Text>
              <ul className="mt-1 flex flex-col gap-0.5">
                {summary.items.map((item) => (
                  <li key={item.id} className="truncate text-sm text-zinc-700">
                    {item.name}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <Dialog.Footer>
            <Button variant="secondary" disabled={isFiring} onClick={onClose}>
              Keep {expertName}
            </Button>
            <Button
              variant="destructive"
              loading={isFiring}
              disabled={isPreviewLoading}
              onClick={handleFire}
              data-testid="fire-expert-confirm"
            >
              Fire {expertName}
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

function FireLine({ children }: PropsWithChildren) {
  return (
    <li className="flex items-start gap-2.5">
      <span
        aria-hidden
        className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-zinc-300"
      />
      <Text variant="body" className="text-zinc-700">
        {children}
      </Text>
    </li>
  );
}
