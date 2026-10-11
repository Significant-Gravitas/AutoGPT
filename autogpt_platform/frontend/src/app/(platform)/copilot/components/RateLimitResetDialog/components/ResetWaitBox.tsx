import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import {
  budgetResetSentence,
  type ResetWait,
} from "@/services/usageExperience/budget";
import { SEND_AT_RESET_LABEL } from "../../SendAtReset/helpers";

interface Reading {
  percent_used: number;
}

interface Props {
  wait: ResetWait;
  /** Off when the dialog's description already says when it resets. */
  showSentence: boolean;
  meters?: { daily?: Reading | null; weekly?: Reading | null } | null;
  onSendAtReset?: () => void;
  isScheduling?: boolean;
}

export function ResetWaitBox({
  wait,
  showSentence,
  meters,
  onSendAtReset,
  isScheduling,
}: Props) {
  return (
    <div className="mb-5 space-y-4 rounded-xl border border-zinc-200 bg-zinc-50 p-4">
      {showSentence && (
        <Text variant="small" className="!leading-5 !text-zinc-600">
          {budgetResetSentence(wait)}
        </Text>
      )}
      <UsageMeters daily={meters?.daily} weekly={meters?.weekly} />
      {onSendAtReset && (
        <Button
          size="small"
          className="w-full"
          onClick={onSendAtReset}
          loading={isScheduling}
        >
          {SEND_AT_RESET_LABEL}
        </Button>
      )}
    </div>
  );
}
