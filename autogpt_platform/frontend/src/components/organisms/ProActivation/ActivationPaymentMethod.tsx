import { useGetV1ManagePaymentMethods } from "@/app/api/__generated__/endpoints/credits/credits";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CreditCardIcon, LinkSquare01Icon } from "@hugeicons/core-free-icons";

export function ActivationPaymentMethod() {
  const { data } = useGetV1ManagePaymentMethods({ query: { retry: false } });
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
        className="flex items-center gap-2 text-sm text-foreground-secondary"
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
        size="small"
        className="h-auto min-w-0 p-0 text-sm underline underline-offset-4"
        aria-label="Manage payment method in a new tab"
      >
        Manage
        <Icon icon={LinkSquare01Icon} size={14} />
      </Button>
    </div>
  );
}
