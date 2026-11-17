import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

interface Props {
  error: unknown;
  onRetry: () => void;
  onSkip: () => void;
}

export function GoogleLookupFailed({ error, onRetry, onSkip }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <ErrorCard
        context="your Google connection"
        responseError={{
          message: error instanceof Error ? error.message : undefined,
        }}
        onRetry={onRetry}
      />
      <Button variant="ghost" onClick={onSkip}>
        Not now
      </Button>
    </div>
  );
}
