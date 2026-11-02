import { getRunStatusInfo } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  Alert01Icon,
  Alert02Icon,
  AlertCircleIcon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  PauseCircleIcon,
  StopCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type StatusIconMap = {
  icon: React.ReactNode;
  colors: string;
};

// oxlint-disable-next-line design-system/no-raw-colors -- yellow-700 #AE8800 has no role; mustard to orange-brown change deferred (see PR)
const PENDING_COLORS = "bg-warning-surface text-yellow-700";

const statusIconMap: Record<AgentExecutionStatus, StatusIconMap> = {
  INCOMPLETE: {
    icon: <Icon icon={AlertCircleIcon} size={16} />,
    colors: "bg-error-surface text-error-foreground",
  },
  QUEUED: {
    icon: <Icon icon={Clock01Icon} size={16} />,
    colors: PENDING_COLORS,
  },
  RUNNING: {
    icon: <Icon icon={PauseCircleIcon} size={16} />,
    colors: PENDING_COLORS,
  },
  REVIEW: {
    icon: <Icon icon={Alert01Icon} size={16} />,
    colors: PENDING_COLORS,
  },
  COMPLETED: {
    icon: <Icon icon={CheckmarkCircle02Icon} size={16} />,
    colors: "bg-success-surface text-success-foreground",
  },
  TERMINATED: {
    icon: <Icon icon={StopCircleIcon} size={16} />,
    colors: "bg-surface-muted text-foreground-body",
  },
  FAILED: {
    icon: <Icon icon={CancelCircleIcon} size={16} />,
    colors: "bg-error-surface text-error-foreground",
  },
};

const completedWithErrorsStyle: StatusIconMap = {
  icon: <Icon icon={Alert02Icon} size={16} />,
  colors: "bg-warning-surface text-warning-foreground",
};

type Props = {
  status: AgentExecutionStatus;
  completedWithErrors?: boolean;
};

export function RunStatusBadge({ status, completedWithErrors = false }: Props) {
  const style = completedWithErrors
    ? completedWithErrorsStyle
    : statusIconMap[status];
  const label = completedWithErrors
    ? "Completed with errors"
    : getRunStatusInfo(status).label;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-md p-1 capitalize",
        style.colors,
      )}
    >
      {style.icon}
      <Text variant="small-medium" tone="inherit">
        {label}
      </Text>
    </div>
  );
}
