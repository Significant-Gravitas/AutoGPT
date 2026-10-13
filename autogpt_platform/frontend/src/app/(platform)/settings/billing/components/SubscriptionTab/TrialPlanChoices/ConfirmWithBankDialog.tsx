import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { formatChoicePrice } from "./helpers";
import type { PaymentPage } from "./useTrialPlanRequest";

interface Props {
  page: PaymentPage | null;
  onClose: () => void;
}

export function ConfirmWithBankDialog({ page, onClose }: Props) {
  return (
    <Dialog
      title="Confirm with your bank"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen: page !== null,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        {page ? (
          <Text variant="body" unmask={false}>
            Your bank wants to confirm the {formatChoicePrice(page.plan)}{" "}
            payment for {page.plan.label}. Pay on Stripe&apos;s secure page and{" "}
            {page.plan.label} starts right away. Until then your trial stays as
            it is.
          </Text>
        ) : null}
        <Dialog.Footer>
          <Button variant="outline" onClick={onClose}>
            Done
          </Button>
          {page ? (
            <Button
              as="NextLink"
              variant="primary"
              href={page.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Pay on Stripe
            </Button>
          ) : null}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
