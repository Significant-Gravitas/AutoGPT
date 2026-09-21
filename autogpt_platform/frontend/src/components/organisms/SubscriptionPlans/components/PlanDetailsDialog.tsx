import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";
import { formatTrialPrice } from "@/components/organisms/TrialCard/helpers";
import { formatPlanAmount, type PlanDialog } from "../helpers";

interface Props {
  kind: PlanDialog;
  setOpen: (open: boolean) => void;
  trialOffer: TrialOfferResponse | null;
  plans: PlanDef[];
}

export function PlanDetailsDialog({ kind, setOpen, trialOffer, plans }: Props) {
  return (
    <Dialog
      title={kind === "compare" ? "Compare plans" : "Your trial. No surprises."}
      variant="compact"
      controlled={{ isOpen: kind !== null, set: setOpen }}
      className="max-w-2xl"
    >
      <Dialog.Content>
        {kind === "compare" ? (
          <PlanComparison plans={plans} />
        ) : (
          trialOffer && <TrialDetails offer={trialOffer} />
        )}
      </Dialog.Content>
    </Dialog>
  );
}

function PlanComparison({ plans }: { plans: PlanDef[] }) {
  return (
    <div className="space-y-4">
      <table className="w-full table-fixed text-left font-sans text-sm text-zinc-800">
        <thead>
          <tr>
            {plans.map((plan) => (
              <th key={plan.key} scope="col" className="pb-3 pr-3 font-medium">
                {plan.name}
                {plan.usage && ` · ${plan.usage} usage`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-zinc-200">
            {plans.map((plan) => (
              <td key={plan.key} className="py-3 pr-3 align-top">
                <ul className="space-y-3">
                  {plan.features.map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
                </ul>
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <Text variant="small" tone="secondary">
        Trial usage is limited. Paid plan allowances apply after the trial.
      </Text>
    </div>
  );
}

function TrialDetails({ offer }: { offer: TrialOfferResponse }) {
  return (
    <div className="space-y-5">
      <Text variant="body" tone="secondary">
        {offer.duration_days} days to put AutoGPT to work.
      </Text>
      <dl className="divide-y divide-zinc-200 border-y border-zinc-200">
        <div className="flex justify-between gap-4 py-4">
          <Text as="dt" variant="body">
            Today
          </Text>
          <Text as="dd" variant="lead-medium" unmask={false}>
            {formatPlanAmount(0, offer.currency)}
          </Text>
        </div>
        <div className="flex justify-between gap-4 py-4">
          <Text as="dt" variant="body">
            After {offer.duration_days} days
          </Text>
          <Text as="dd" variant="lead-medium" unmask={false}>
            {formatTrialPrice(offer)}
          </Text>
        </div>
      </dl>
      <Text variant="body" tone="secondary">
        Card required. No subscription charge today. Your paid subscription
        starts automatically after the trial, plus applicable tax, unless you
        cancel before it ends.
      </Text>
      <Text variant="body" tone="secondary">
        Trial usage is limited. Canceling ends trial access immediately. You can
        manage or cancel your plan in billing.
      </Text>
    </div>
  );
}
