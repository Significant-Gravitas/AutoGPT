import { Button } from "@/components/atoms/Button/Button";
import type { ComponentProps } from "react";

import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import { SEND_AT_RESET_LABEL } from "../../SendAtReset/helpers";
import { LimitReachedExtras } from "../../UsageLimits/PaidReset/LimitReachedExtras";

interface Reading {
  percent_used: number;
}

type Extras = NonNullable<ComponentProps<typeof LimitReachedExtras>["usage"]>;

interface Props {
  meters?:
    | ({ daily?: Reading | null; weekly?: Reading | null } & Extras)
    | null;
  /** The limit being waited on. */
  window?: "daily" | "weekly";
  onSendAtReset?: () => void;
  isScheduling?: boolean;
  onBought?: () => void;
}

export function ResetWaitBox({
  meters,
  window = "daily",
  onSendAtReset,
  isScheduling,
  onBought,
}: Props) {
  return (
    <div className="mb-5 space-y-4 rounded-xl border border-border-base bg-surface-muted p-4">
      <UsageMeters daily={meters?.daily} weekly={meters?.weekly} />
      <LimitReachedExtras window={window} usage={meters} onBought={onBought} />
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
