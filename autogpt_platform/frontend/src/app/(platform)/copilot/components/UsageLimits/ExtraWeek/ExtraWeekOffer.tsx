"use client";

import type { ExtraWeekOffer as Offer } from "@/app/api/__generated__/models/extraWeekOffer";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EXTRA_WEEK_TERMS, formatExtraWeekPrice } from "./helpers";
import { useExtraWeekPurchase } from "./useExtraWeekPurchase";

interface Props {
  offer: Offer;
  label?: string;
  variant?: "primary" | "secondary";
  onBought?: () => void;
}

export function ExtraWeekOffer({
  offer,
  label = "Add an extra week",
  variant = "secondary",
  onBought,
}: Props) {
  const purchase = useExtraWeekPurchase({ onBought });
  const price = formatExtraWeekPrice(offer.price_credits);
  return (
    <>
      <Button
        type="button"
        size="small"
        variant={variant}
        onClick={purchase.open}
      >
        {label} · {price}
      </Button>
      <Dialog
        title="Add an extra week of usage?"
        styling={{ maxWidth: "440px" }}
        controlled={{ isOpen: purchase.isOpen, set: purchase.setIsOpen }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-2">
            <Text variant="body" tone="body">
              {price} from your AutoGPT credit balance buys one more week of
              your plan’s usage.
            </Text>
            <ul className="flex list-disc flex-col gap-1 pl-5">
              {EXTRA_WEEK_TERMS.map((term) => (
                <li key={term}>
                  <Text variant="body" as="span" tone="body">
                    {term}
                  </Text>
                </li>
              ))}
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
              Pay {price}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
