import type { AutoTopUpConfig } from "@/app/api/__generated__/models/autoTopUpConfig";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CreditCardIcon } from "@hugeicons/core-free-icons";
import { getTrialCardNote } from "../helpers";

interface Props {
  id: string;
  trial: TrialStatusResponse | undefined;
  autoRefill: AutoTopUpConfig | undefined;
}

export function WalletTrialNote({ id, trial, autoRefill }: Props) {
  const note = getTrialCardNote(trial, autoRefill);
  if (!note) return null;

  return (
    <div
      id={id}
      className="mx-5 mt-4 flex items-start gap-3 rounded-large bg-surface-muted px-3 py-2.5"
    >
      <Icon
        icon={CreditCardIcon}
        size={18}
        aria-hidden
        className="mt-0.5 shrink-0 text-foreground-tertiary"
      />
      <div className="flex min-w-0 flex-col gap-0.5">
        <Text variant="body-medium" tone="primary" className="text-pretty">
          {note.title}
        </Text>
        <Text variant="small" tone="secondary">
          {note.body}
        </Text>
      </div>
    </div>
  );
}
