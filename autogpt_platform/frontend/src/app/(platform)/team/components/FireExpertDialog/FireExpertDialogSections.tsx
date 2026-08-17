import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ReactNode } from "react";
import type { PauseItem } from "./helpers";

type PreviewProps = {
  expertName: string;
  automationLine: string;
  items: PauseItem[];
  isLoading: boolean;
  isError: boolean;
  isReady: boolean;
  onRetry: () => unknown;
};

type FooterProps = {
  expertName: string;
  isFiring: boolean;
  isPreviewLoading: boolean;
  onClose: () => void;
  onFire: () => void;
};

type PauseItemsPanelProps = {
  items: PauseItem[];
};

type FireLineProps = {
  children: ReactNode;
};

export function FireExpertPreview({
  expertName,
  automationLine,
  items,
  isLoading,
  isError,
  isReady,
  onRetry,
}: PreviewProps) {
  function getAutomationLineText() {
    if (isLoading) return "Checking what will pause…";
    if (isError)
      return "We couldn't preview what pauses, but you can still fire them.";
    return automationLine;
  }

  function handleRetry() {
    void onRetry();
  }

  return (
    <>
      <Text variant="body" className="text-zinc-600">
        Here is exactly what happens when you fire {expertName}.
      </Text>
      <ul className="flex flex-col gap-2.5">
        <FireLine>Installed workflows stay in your library.</FireLine>
        <FireLine>{getAutomationLineText()}</FireLine>
        <FireLine>Any chat history stays available but read-only.</FireLine>
        <FireLine>Their work stays yours.</FireLine>
      </ul>
      {isError ? (
        <div className="flex justify-end">
          <Button
            variant="secondary"
            size="small"
            onClick={handleRetry}
            data-testid="fire-preview-retry"
          >
            Retry preview
          </Button>
        </div>
      ) : null}
      {isReady && items.length > 0 ? <PauseItemsPanel items={items} /> : null}
    </>
  );
}

function PauseItemsPanel({ items }: PauseItemsPanelProps) {
  return (
    <div className="rounded-xl bg-zinc-50 px-3.5 py-3 ring-1 ring-inset ring-zinc-200/80">
      <Text variant="small" className="text-zinc-500">
        Pausing now
      </Text>
      <ul className="mt-1 flex flex-col gap-0.5">
        {items.map((item) => (
          <li key={item.id} className="truncate text-sm text-zinc-700">
            {item.name}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function FireExpertFooter({
  expertName,
  isFiring,
  isPreviewLoading,
  onClose,
  onFire,
}: FooterProps) {
  return (
    <Dialog.Footer>
      <Button variant="secondary" disabled={isFiring} onClick={onClose}>
        Keep {expertName}
      </Button>
      <Button
        variant="destructive"
        loading={isFiring}
        disabled={isPreviewLoading}
        onClick={onFire}
        data-testid="fire-expert-confirm"
      >
        Fire {expertName}
      </Button>
    </Dialog.Footer>
  );
}

function FireLine({ children }: FireLineProps) {
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
