import { Text } from "@/components/atoms/Text/Text";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";

interface Props {
  plans: PlanDef[];
}

export function PlanComparison({ plans }: Props) {
  return (
    <div data-slot="plan-comparison" className="space-y-4">
      <table
        data-slot="plan-comparison-table"
        className="w-full table-fixed text-left font-sans text-sm text-foreground-emphasis"
      >
        <thead data-slot="plan-comparison-head">
          <tr data-slot="plan-comparison-head-row">
            {plans.map((plan) => (
              <th
                key={plan.key}
                scope="col"
                data-slot="plan-comparison-column"
                className="pb-3 pr-3 font-medium"
              >
                {plan.name}
                {plan.usage && ` · ${plan.usage} usage`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody data-slot="plan-comparison-body">
          <tr
            data-slot="plan-comparison-row"
            className="border-t border-border-base"
          >
            {plans.map((plan) => (
              <td
                key={plan.key}
                data-slot="plan-comparison-cell"
                className="py-3 pr-3 align-top"
              >
                <ul data-slot="plan-comparison-features" className="space-y-3">
                  {plan.features.map((feature) => (
                    <li key={feature} data-slot="plan-comparison-feature">
                      {feature}
                    </li>
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
