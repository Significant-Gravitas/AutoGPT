import { Button } from "@/components/atoms/Button/Button";
import type { ComponentProps } from "react";

import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import { SEND_AT_RESET_LABEL } from "../../SendAtReset/helpers";
import { WeeklyLimitExtras } from "../../UsageLimits/ExtraWeek/WeeklyLimitExtras";

interface Reading {
  percent_used: number;
}

type Extras = NonNullable<ComponentProps<typeof WeeklyLimitExtras>["usage"]>;

interface Props {
  meters?:
    | ({ daily?: Reading | null; weekly?: Reading | null } & Extras)
    | null;
  /** The limit being waited on; the weekly one can be extended. */
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
      <WeeklyLimitExtras window={window} usage={meters} onBought={onBought} />
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
