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
    <ol data-slot="activation-timeline" className="space-y-4 text-sm">
      <li
        data-slot="activation-timeline-item"
        className="relative grid grid-cols-[0.75rem_4.5rem_1fr] gap-x-3 before:absolute before:bottom-[-1rem] before:left-[5px] before:top-4 before:w-px before:bg-surface-active"
      >
        <span
          aria-hidden="true"
          data-slot="activation-timeline-dot"
          className="mt-1.5 h-2.5 w-2.5 rounded-full bg-action-hover"
        />
        <span
          data-slot="activation-timeline-date"
          className="font-medium text-foreground-strong"
        >
          Today
        </span>
        <span
          data-slot="activation-timeline-description"
          className="leading-relaxed text-foreground-secondary"
        >
          After payment succeeds, {plan} starts and your daily and weekly
          allowances reset.
        </span>
      </li>
      <li
        data-slot="activation-timeline-item"
        className="grid grid-cols-[0.75rem_4.5rem_1fr] gap-x-3"
      >
        <span
          aria-hidden="true"
          data-slot="activation-timeline-dot"
          // TODO(tokens): no border role for zinc-400 (border-strong is
          // zinc-300).
          className="mt-1.5 h-2.5 w-2.5 rounded-full border border-zinc-400 bg-surface"
        />
        <span
          data-slot="activation-timeline-date"
          className="font-medium text-foreground-strong"
        >
          {terms.next_payment_at ? shortDate(terms.next_payment_at) : "Renewal"}
        </span>
        <span
          data-slot="activation-timeline-description"
          className="leading-relaxed text-foreground-secondary"
        >
          {nextPaymentDescription(terms)}
        </span>
      </li>
    </ol>
  );
}
