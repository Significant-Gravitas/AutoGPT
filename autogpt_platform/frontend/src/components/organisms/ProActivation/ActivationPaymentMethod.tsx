import {
  getGetV1ManagePaymentMethodsQueryKey,
  useGetV1ManagePaymentMethods,
} from "@/app/api/__generated__/endpoints/credits/credits";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { getPersonalBillingQueryOptions } from "@/services/org-team/org-wallet";
import { useActiveOrgWallet } from "@/services/org-team/useActiveOrgWallet";
import { CreditCardIcon, LinkSquare01Icon } from "@hugeicons/core-free-icons";

export function ActivationPaymentMethod() {
  // A personal subscription is the person's own, whichever organization they
  // work in now, so its payment method is too.
  const wallet = useActiveOrgWallet();
  const personal = getPersonalBillingQueryOptions(
    getGetV1ManagePaymentMethodsQueryKey(),
    wallet,
  );
  const { data } = useGetV1ManagePaymentMethods({
    request: personal.request,
    query: { ...personal.query, retry: false },
  });
  const raw = data?.data;
  const portal =
    raw && typeof raw === "object" && "url" in raw ? raw.url : null;
  const url =
    typeof portal === "string" &&
    /^https:\/\/billing\.stripe\.com\//.test(portal)
      ? portal
      : null;
  return (
    <div
      data-slot="activation-payment-method"
      className="flex items-center justify-between gap-3 rounded-large border border-border-base px-4 py-3"
    >
      <div
        data-slot="activation-payment-method-label"
        className="flex items-center gap-2 text-body text-foreground-secondary"
      >
        <Icon icon={CreditCardIcon} size={18} />
        Payment method
      </div>
      <Button
        as="NextLink"
        href={url ?? "/settings/billing"}
        target="_blank"
        rel="noopener noreferrer"
        variant="ghost"
        size="inline"
        className="underline underline-offset-4"
        aria-label="Manage payment method in a new tab"
      >
        Manage
        <Icon icon={LinkSquare01Icon} size={14} />
      </Button>
    </div>
  );
}
