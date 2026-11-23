import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { LISTING_CHECK_FAILED_MESSAGE } from "../helpers";
import { StepFooter } from "./StepFooter";

export function ListingCheckPending() {
  return (
    <div
      role="status"
      aria-label="Checking this workflow's Marketplace listing"
      className="flex w-full flex-col gap-4 px-1 pb-5 sm:px-2"
    >
      <Skeleton className="h-6 w-1/2" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
    </div>
  );
}

interface Props {
  onRetry: () => void;
  onCancel: () => void;
}

export function ListingCheckFailed({ onRetry, onCancel }: Props) {
  return (
    <div className="mx-auto flex w-full flex-col">
      <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-error/10 bg-error-surface px-6 py-10 text-center">
        <Icon icon={AlertCircleIcon} size={32} tone="danger" />
        <Text variant="body" tone="error-strong" role="alert">
          {LISTING_CHECK_FAILED_MESSAGE}
        </Text>
      </div>
      <StepFooter
        secondary={
          <Button
            variant="secondary"
            size="small"
            onClick={onCancel}
            className="w-full sm:w-auto"
          >
            Cancel
          </Button>
        }
        primary={
          <Button size="small" onClick={onRetry} className="w-full sm:w-auto">
            Try again
          </Button>
        }
      />
    </div>
  );
}
