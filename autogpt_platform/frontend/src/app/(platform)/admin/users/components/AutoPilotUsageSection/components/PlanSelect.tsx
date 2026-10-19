import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { getPlanName } from "../../../helpers";
import { getPlanOptions, isAssignableTier } from "../helpers";

interface Props {
  tier: string;
  multipliers?: Record<string, number>;
  disabled: boolean;
  onPick: (tier: string) => void;
}

export function PlanSelect({ tier, multipliers, disabled, onPick }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <Text variant="small-medium" as="span" aria-hidden>
        Plan
      </Text>
      <Select
        id="user-plan"
        label="Plan"
        hideLabel
        size="small"
        value={isAssignableTier(tier) ? tier : undefined}
        placeholder={`${getPlanName(tier)} (set by billing)`}
        options={getPlanOptions(multipliers)}
        onValueChange={(picked) => {
          if (picked !== tier) onPick(picked);
        }}
        disabled={disabled}
        wrapperClassName="mb-0 w-full sm:w-[260px]"
      />
    </div>
  );
}
