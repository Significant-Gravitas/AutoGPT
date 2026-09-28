import type { DelegationSummaryStatus } from "@/app/api/__generated__/models/delegationSummaryStatus";
import { Badge } from "@/components/atoms/Badge/Badge";
import { type BadgeTone, getDelegationBadge } from "./helpers";

interface Props {
  status: DelegationSummaryStatus;
  label?: string;
}

const WORKING_CLASS = "bg-violet-50 text-violet-700 ring-violet-600/15";

function toBadgeVariant(tone: BadgeTone) {
  return tone === "working" ? "info" : tone;
}

export function DelegationStatusBadge({ status, label }: Props) {
  const badge = getDelegationBadge(status);
  return (
    <Badge
      variant={toBadgeVariant(badge.tone)}
      className={badge.tone === "working" ? WORKING_CLASS : undefined}
    >
      {label ?? badge.label}
    </Badge>
  );
}

export function WorkingForPill({ children }: { children: React.ReactNode }) {
  return (
    <Badge variant="info" className={WORKING_CLASS}>
      {children}
    </Badge>
  );
}
