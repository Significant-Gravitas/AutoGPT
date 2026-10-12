import type { ActivationTerms } from "@/app/api/__generated__/models/activationTerms";
import {
  nextPaymentDescription,
  shortDate,
} from "@/services/pro-activation/reviewDetails";

interface Props {
  terms: ActivationTerms;
}

export function ActivationTimeline({ terms }: Props) {
  const plan = terms.plan === "MAX" ? "Max" : "Pro";
  return (
    <ol className="space-y-4 text-sm">
      <li className="relative grid grid-cols-[0.75rem_4.5rem_1fr] gap-x-3 before:absolute before:bottom-[-1rem] before:left-[5px] before:top-4 before:w-px before:bg-zinc-200">
        <span
          aria-hidden="true"
          className="mt-1.5 h-2.5 w-2.5 rounded-full bg-zinc-900"
        />
        <span className="font-medium text-zinc-900">Today</span>
        <span className="leading-relaxed text-zinc-600">
          After payment succeeds, {plan} starts and your daily and weekly
          allowances reset.
        </span>
      </li>
      <li className="grid grid-cols-[0.75rem_4.5rem_1fr] gap-x-3">
        <span
          aria-hidden="true"
          className="mt-1.5 h-2.5 w-2.5 rounded-full border border-zinc-400 bg-white"
        />
        <span className="font-medium text-zinc-900">
          {terms.next_payment_at ? shortDate(terms.next_payment_at) : "Renewal"}
        </span>
        <span className="leading-relaxed text-zinc-600">
          {nextPaymentDescription(terms)}
        </span>
      </li>
    </ol>
  );
}
