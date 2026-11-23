import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";

/** Long enough to read the reason and reach Retry. */
const FAILED_TOAST_MS = 10_000;

interface Args {
  owner: string;
  error: unknown;
  onRetry: () => void;
}

/** The one message when a task was saved but did not start: why, and a
 *  Retry that starts it again. */
export function showStartFailed({ owner, error, onRetry }: Args) {
  toast({
    title: `Saved for ${owner}, but couldn't start`,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
    duration: FAILED_TOAST_MS,
    action: (
      <Button type="button" variant="secondary" size="xs" onClick={onRetry}>
        Retry
      </Button>
    ),
  });
}
