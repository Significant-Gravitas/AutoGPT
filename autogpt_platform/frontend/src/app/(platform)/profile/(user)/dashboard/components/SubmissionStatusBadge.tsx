import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { Badge } from "@/components/atoms/Badge/Badge";

interface Props {
  status: SubmissionStatus;
}

const STATUS_INFO: Record<
  SubmissionStatus,
  { label: string; variant: "success" | "error" | "warning" | "info" }
> = {
  [SubmissionStatus.DRAFT]: { label: "Draft", variant: "info" },
  [SubmissionStatus.PENDING]: { label: "Awaiting review", variant: "warning" },
  [SubmissionStatus.APPROVED]: { label: "Approved", variant: "success" },
  [SubmissionStatus.REJECTED]: { label: "Rejected", variant: "error" },
};

export function SubmissionStatusBadge({ status }: Props) {
  const info = STATUS_INFO[status] ?? STATUS_INFO[SubmissionStatus.PENDING];
  return <Badge variant={info.variant}>{info.label}</Badge>;
}
