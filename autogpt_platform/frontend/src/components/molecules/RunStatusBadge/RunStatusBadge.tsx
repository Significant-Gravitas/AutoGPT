import { Badge } from "@/components/atoms/Badge/Badge";

type BadgeVariant = "success" | "error" | "warning" | "info";

interface RunStatusInfo {
  label: string;
  variant: BadgeVariant;
}

interface Props {
  status: string;
}

const STATUS_INFO: Record<string, RunStatusInfo> = {
  COMPLETED: { label: "Completed", variant: "success" },
  FAILED: { label: "Failed", variant: "error" },
  RUNNING: { label: "Running", variant: "info" },
  QUEUED: { label: "Queued", variant: "info" },
  REVIEW: { label: "Waiting for review", variant: "warning" },
  TERMINATED: { label: "Stopped", variant: "info" },
  INCOMPLETE: { label: "Incomplete", variant: "info" },
};

export function getRunStatusInfo(status: string): RunStatusInfo {
  return (
    STATUS_INFO[status.toUpperCase()] ?? {
      label: status,
      variant: "info",
    }
  );
}

export function RunStatusBadge({ status }: Props) {
  const info = getRunStatusInfo(status);
  return (
    <Badge variant={info.variant} size="small">
      {info.label}
    </Badge>
  );
}
