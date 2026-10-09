import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import {
  describeTrialTimeLeft,
  formatTrialDays,
  formatTrialEnd,
} from "../helpers";

interface Props {
  endsAt: TrialStatusResponse["ends_at"];
}

export function TrialTimeLeft({ endsAt }: Props) {
  if (!endsAt) return <>You still have full access until your trial ends.</>;
  const left = describeTrialTimeLeft(endsAt);
  if (left.kind !== "days")
    return (
      <>
        You still have full access until{" "}
        <strong className="font-semibold">
          {left.time} {left.kind}
        </strong>
        .
      </>
    );
  return (
    <>
      You still have{" "}
      <strong className="font-semibold">{formatTrialDays(left.days)}</strong> of
      full access, until{" "}
      <strong className="font-semibold">{formatTrialEnd(endsAt)}</strong>.
    </>
  );
}
