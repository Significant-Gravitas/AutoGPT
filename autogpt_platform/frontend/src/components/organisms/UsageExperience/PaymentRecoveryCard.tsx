import { CreditCardIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { formatPaymentDue } from "@/services/usageExperience/payment";
import { PaymentDueAction } from "./PaymentDueAction";

interface Props {
  payment: ReturnType<typeof formatPaymentDue>;
  fallback: string;
}

export function PaymentRecoveryCard({ payment, fallback }: Props) {
  return (
    <section
      aria-label="Payment"
      className="w-full rounded-2xl border border-zinc-200 bg-white p-6 text-left"
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 items-center justify-center rounded-lg bg-white text-zinc-600 ring-1 ring-zinc-200">
          <Icon icon={CreditCardIcon} size={17} />
        </span>
        <Text variant="small-medium" className="text-zinc-600">
          Payment
        </Text>
      </div>
      <Text
        variant="h4"
        as="h2"
        className="mb-3 mt-4 text-[23px] leading-8 tracking-[-0.04em]"
      >
        Your payment didn’t go through
      </Text>
      <Text
        variant="small"
        tone="secondary"
        className="max-w-[38rem] leading-6"
      >
        {payment?.description ?? fallback}
      </Text>
      {payment ? (
        <div className="mt-5">
          <PaymentDueAction
            url={payment.url}
            label={payment.actionLabel}
            layout="inline"
          />
        </div>
      ) : null}
    </section>
  );
}
