"use client";

import {
  ArrowRightIcon,
  NotePencilIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { StepFooter } from "./StepFooter";

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
                leftIcon={<NotePencilIcon size={14} weight="bold" />}
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
              rightIcon={<StorefrontIcon size={14} weight="bold" />}
              data-testid="view-marketplace-button"
            >
              View on marketplace
            </Button>
          ) : isRejected || isDraft || isDashboardPage ? null : (
            <Button
              size="small"
              onClick={onViewProgress}
              className="w-full sm:w-auto"
              rightIcon={<ArrowRightIcon size={14} weight="bold" />}
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
