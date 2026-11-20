"use client";
import { Button } from "@/components/atoms/Button/Button";
import { StepFooter } from "./StepFooter";
import {
  ArrowRight02Icon,
  NoteEditIcon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onDone: () => void;
  onViewProgress: () => void;
  onEdit?: () => void;
  isApproved: boolean;
  isRejected: boolean;
  isDraft: boolean;
  isPending: boolean;
  isDashboardPage: boolean;
  marketplaceUrl?: string;
}

export function ReviewStepFooter({
  onDone,
  onViewProgress,
  onEdit,
  isApproved,
  isRejected,
  isDraft,
  isPending,
  isDashboardPage,
  marketplaceUrl,
}: Props) {
  return (
    <div className="mt-8 w-full">
      <StepFooter
        secondary={
          <>
            {onEdit && isPending ? (
              <Button
                variant="ghost"
                size="small"
                onClick={onEdit}
                className="w-full sm:w-auto"
                leftIcon={<Icon icon={NoteEditIcon} size={14} />}
                data-testid="edit-submission-button"
              >
                Edit details
              </Button>
            ) : null}
            <Button
              variant="secondary"
              size="small"
              onClick={onDone}
              className="w-full sm:w-auto"
            >
              Done
            </Button>
          </>
        }
        primary={
          isApproved && marketplaceUrl ? (
            <Button
              as="NextLink"
              href={marketplaceUrl}
              size="small"
              className="w-full sm:w-auto"
              rightIcon={<Icon icon={Store01Icon} size={14} />}
              data-testid="view-marketplace-button"
            >
              View on marketplace
            </Button>
          ) : isRejected || isDraft || isDashboardPage ? null : (
            <Button
              size="small"
              onClick={onViewProgress}
              className="w-full sm:w-auto"
              rightIcon={<Icon icon={ArrowRight02Icon} size={14} />}
              data-testid="view-progress-button"
            >
              Go to Creator Dashboard
            </Button>
          )
        }
      />
    </div>
  );
}
