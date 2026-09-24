import { Button } from "@/components/atoms/Button/Button";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";

interface Props {
  count: number;
  confirming: boolean;
  busy: boolean;
  onAsk: () => void;
  onCancel: () => void;
  onConfirm: () => void;
}

export function RejectAllFooter({
  count,
  confirming,
  busy,
  onAsk,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-zinc-100 bg-zinc-50/60 px-4 py-2 text-sm">
      {confirming ? (
        <>
          <span role="alert" className="text-zinc-700">
            Reject all {count}? {AUTOPILOT_NAME} will be told none of them ran.
          </span>
          <Button
            size="xs"
            variant="primary"
            className="rounded-full"
            disabled={busy}
            onClick={onConfirm}
          >
            Reject all
          </Button>
          <Button
            size="xs"
            variant="ghost"
            className="rounded-full"
            onClick={onCancel}
          >
            Cancel
          </Button>
        </>
      ) : (
        <Button
          variant="link"
          size="small"
          className="text-sm"
          disabled={busy}
          onClick={onAsk}
        >
          Reject all {count}…
        </Button>
      )}
    </div>
  );
}
