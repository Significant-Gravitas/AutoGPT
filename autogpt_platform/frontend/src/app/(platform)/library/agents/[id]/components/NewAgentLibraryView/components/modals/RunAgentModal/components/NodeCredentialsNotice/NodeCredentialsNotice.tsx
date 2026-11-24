import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";

interface Props {
  isRetrying: boolean;
  onRetry: () => void;
}

export function NodeCredentialsNotice({ isRetrying, onRetry }: Props) {
  return (
    <Alert variant="warning" aria-live="polite">
      <AlertDescription>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text variant="body">
            Couldn&apos;t load the credentials this workflow&apos;s steps use.
            Pick one below, or try again.
          </Text>
          <Button
            variant="secondary"
            size="small"
            loading={isRetrying}
            disabled={isRetrying}
            onClick={onRetry}
          >
            Try again
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}
