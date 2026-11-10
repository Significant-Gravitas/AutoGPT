import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  expertName: string;
  isRetrying: boolean;
  onClose: () => void;
  onRetry: () => void;
}

export function ResetBusy({ expertName, isRetrying, onClose, onRetry }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" tone="secondary">
        {expertName} is in the middle of something, so nothing was reset. Try
        again once they finish.
      </Text>
      <Dialog.Footer>
        <Button
          size="small"
          variant="secondary"
          disabled={isRetrying}
          onClick={onClose}
        >
          Close
        </Button>
        <Button
          size="small"
          variant="primary"
          loading={isRetrying}
          onClick={onRetry}
          data-testid="reset-expert-retry"
        >
          Try again
        </Button>
      </Dialog.Footer>
    </div>
  );
}
