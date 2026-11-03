import { Button } from "@/components/atoms/Button/Button";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";

interface Props {
  rejectAll: {
    count: number;
    confirming: boolean;
    onAsk: () => void;
    onCancel: () => void;
    onConfirm: () => void;
  } | null;
  // Shown once something is answered and something still waits.
  onContinue: (() => void) | null;
  busy: boolean;
}

export function QueueFooter({ rejectAll, onContinue, busy }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-separator bg-surface-muted/60 px-4 py-2 text-body">
      {rejectAll?.confirming ? (
        <>
          <span role="alert" className="text-foreground-body">
            Reject all {rejectAll.count}? {AUTOPILOT_NAME} will be told none of
            them ran.
          </span>
          <Button
            size="xs"
            variant="primary"
            shape="pill"
            disabled={busy}
            onClick={rejectAll.onConfirm}
          >
            Reject all
          </Button>
          <Button
            size="xs"
            variant="ghost"
            shape="pill"
            onClick={rejectAll.onCancel}
          >
            Cancel
          </Button>
        </>
      ) : rejectAll ? (
        <Button
          variant="link"
          size="inline"
          disabled={busy}
          onClick={rejectAll.onAsk}
        >
          Reject all {rejectAll.count}…
        </Button>
      ) : null}
      {onContinue && !rejectAll?.confirming && (
        <Button
          type="button"
          size="small"
          variant="secondary"
          className="ml-auto min-w-0"
          disabled={busy}
          onClick={onContinue}
        >
          Close and continue
        </Button>
      )}
    </div>
  );
}
