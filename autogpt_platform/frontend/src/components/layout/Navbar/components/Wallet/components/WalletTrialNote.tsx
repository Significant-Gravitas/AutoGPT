import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CreditCardIcon } from "@hugeicons/core-free-icons";
import { getTrialPlanLine } from "../helpers";

interface Props {
  id: string;
  trial: TrialStatusResponse | undefined;
}

export function WalletTrialNote({ id, trial }: Props) {
  const line = getTrialPlanLine(trial);
  if (!line) return null;

  return (
    <div id={id} className="flex items-center gap-2 px-5 pt-4">
      <Icon
        icon={CreditCardIcon}
        size={16}
        aria-hidden
        className="shrink-0 text-foreground-tertiary"
      />
      <Text variant="small" tone="secondary">
        {line}
      </Text>
    </div>
  );
}
