import { SubmissionStatus as Status } from "@/app/api/__generated__/models/submissionStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  status: Status;
}

const statusConfig: Record<
  Status,
  { fill: string; dot: string; label: string }
> = {
  [Status.DRAFT]: { fill: "bg-info-surface", dot: "bg-info", label: "Draft" },
  [Status.PENDING]: {
    fill: "bg-warning-surface",
    dot: "bg-warning",
    label: "Awaiting review",
  },
  [Status.APPROVED]: {
    fill: "bg-success-surface",
    dot: "bg-success",
    label: "Approved",
  },
  [Status.REJECTED]: {
    fill: "bg-error-surface",
    dot: "bg-error",
    label: "Rejected",
  },
};

export function SubmissionStatus({ status }: Props) {
  const config = statusConfig[status] ?? statusConfig[Status.PENDING];

  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1",
        config.fill,
      )}
    >
      <div className={cn("h-3 w-3 rounded-full", config.dot)} />
      <Text variant="body-tight" as="div" tone="secondary">
        {config.label}
      </Text>
    </div>
  );
}
