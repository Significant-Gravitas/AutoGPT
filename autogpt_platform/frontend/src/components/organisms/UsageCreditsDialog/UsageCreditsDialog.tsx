"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useUsageCreditsDialog } from "./useUsageCreditsDialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  /** Off on the billing page itself, where the link would go nowhere. */
  showBillingLink?: boolean;
}

export function UsageCreditsDialog({
  isOpen,
  onOpenChange,
  showBillingLink = true,
}: Props) {
  return (
    <Dialog
      title="How usage and credits work"
      variant="compact"
      controlled={{ isOpen, set: onOpenChange }}
      className="max-w-xl"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-5">
          <UsageCreditsAnswers />
          {showBillingLink && (
            <div className="flex justify-end">
              <Button
                as="NextLink"
                href="/settings/billing"
                variant="secondary"
                size="small"
                rightIcon={
                  <Icon icon={ArrowRight01Icon} size={16} aria-hidden />
                }
                onClick={() => onOpenChange(false)}
              >
                Go to billing settings
              </Button>
            </div>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

// Mounted only while the dialog is open, so its usage queries run on demand.
function UsageCreditsAnswers() {
  const { answers } = useUsageCreditsDialog();
  return (
    <dl className="flex flex-col gap-4">
      {answers.map((item) => (
        <div key={item.question} className="flex flex-col gap-1">
          <Text as="dt" variant="body-medium" tone="primary">
            {item.question}
          </Text>
          <Text as="dd" variant="body" tone="body">
            {item.answer}
          </Text>
        </div>
      ))}
    </dl>
  );
}
