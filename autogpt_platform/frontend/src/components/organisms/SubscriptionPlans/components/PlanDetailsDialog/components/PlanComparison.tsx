import { Text } from "@/components/atoms/Text/Text";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";

interface Props {
  plans: PlanDef[];
}

export function PlanComparison({ plans }: Props) {
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
