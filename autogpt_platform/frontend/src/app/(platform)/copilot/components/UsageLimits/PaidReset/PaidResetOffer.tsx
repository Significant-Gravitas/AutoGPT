"use client";

import type { UsageResetOffer as Offer } from "@/app/api/__generated__/models/usageResetOffer";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { formatExactReset } from "@/services/usageExperience/resetTime";
import {
  describePayment,
  formatResetPrice,
  payLabel,
  resetPromise,
} from "./helpers";
import { usePaidResetPurchase } from "./usePaidResetPurchase";

interface Props {
  offer: Offer;
  /** The weekly reset, which a paid reset lasts until and does not move. */
  weeklyResetsAt?: string | Date | null;
  variant?: "primary" | "secondary";
  onBought?: () => void;
  goToCheckout?: (url: string) => void;
}

export function PaidResetOffer({
  offer,
  weeklyResetsAt,
  variant = "secondary",
  onBought,
  goToCheckout,
}: Props) {
  const purchase = usePaidResetPurchase({
    priceCredits: offer.price_credits,
    onBought,
    goToCheckout,
  });
  const resetsAt = useReadEveryRender(() => formatExactReset(weeklyResetsAt));
  const price = formatResetPrice(offer.price_credits);
  const payWith = offer.pay_with ?? "card";
  return (
    <>
      <Button
        type="button"
        size="small"
        variant={variant}
        onClick={purchase.open}
      >
        Reset limits now · {price}
      </Button>
      <Dialog
        title="Reset your limits now?"
        styling={{ maxWidth: "440px" }}
        controlled={{ isOpen: purchase.isOpen, set: purchase.setIsOpen }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-2">
            <Text variant="body" tone="body">
              {resetPromise(resetsAt)}
            </Text>
            <ul className="flex list-disc flex-col gap-1 pl-5">
              <li>
                <Text variant="body" as="span" tone="body">
                  Your weekly reset doesn’t move. Usage you haven’t used by then
                  doesn’t carry over.
                </Text>
              </li>
              <li>
                <Text variant="body" as="span" tone="body">
                  {describePayment(payWith, price)}
                </Text>
              </li>
            </ul>
          </div>
          <Dialog.Footer>
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => purchase.setIsOpen(false)}
              disabled={purchase.isBuying}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="primary"
              size="small"
              onClick={() => void purchase.confirm()}
              disabled={purchase.isBuying}
              loading={purchase.isBuying}
            >
              {payLabel(payWith, price)}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
