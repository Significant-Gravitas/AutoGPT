import { Button } from "@/components/atoms/Button/Button";
import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import { SEND_AT_RESET_LABEL } from "../../SendAtReset/helpers";

interface Reading {
  percent_used: number;
}

interface Props {
  meters?: { daily?: Reading | null; weekly?: Reading | null } | null;
  onSendAtReset?: () => void;
  isScheduling?: boolean;
}

export function ResetWaitBox({ meters, onSendAtReset, isScheduling }: Props) {
  return (
    <div className="mb-5 space-y-4 rounded-xl border border-zinc-200 bg-zinc-50 p-4">
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
