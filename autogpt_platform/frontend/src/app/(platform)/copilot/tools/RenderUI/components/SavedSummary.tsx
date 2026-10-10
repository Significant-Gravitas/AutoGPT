import { Button } from "@/components/atoms/Button/Button";

interface Props {
  summary: string;
  valid: boolean;
  locked: boolean;
  isSending: boolean;
  onRebuild: () => void;
}

export function SavedSummary({
  summary,
  valid,
  locked,
  isSending,
  onRebuild,
}: Props) {
  return (
    <div className="space-y-4">
      {!valid && (
        <p className="text-sm text-zinc-500">
          The interactive view couldn’t be displayed.{" "}
          {summary
            ? "Here’s the saved summary."
            : "The response may have been interrupted."}
        </p>
      )}
      {summary && (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-800">
          {summary}
        </p>
      )}
      {!valid && !locked && (
        <Button
          size="small"
          variant="secondary"
          disabled={isSending}
          onClick={onRebuild}
        >
          Rebuild this view
        </Button>
      )}
    </div>
  );
}
