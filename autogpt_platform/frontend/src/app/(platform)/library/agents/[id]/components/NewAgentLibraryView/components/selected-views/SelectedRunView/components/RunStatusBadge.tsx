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
  bgColor: string;
  textColor: string;
};

const statusIconMap: Record<AgentExecutionStatus, StatusIconMap> = {
  INCOMPLETE: {
    icon: (
      <Icon
        icon={AlertCircleIcon}
        size={16}
        className="text-error-foreground"
      />
    ),
    bgColor: "bg-error-surface",
    textColor: "!text-error-foreground",
  },
  QUEUED: {
    // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
    icon: <Icon icon={Clock01Icon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  RUNNING: {
    // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
    icon: <Icon icon={PauseCircleIcon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  REVIEW: {
    // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
    icon: <Icon icon={Alert01Icon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  COMPLETED: {
    icon: (
      <Icon
        icon={CheckmarkCircle02Icon}
        size={16}
        className="text-success-foreground"
      />
    ),
    bgColor: "bg-success-surface",
    textColor: "!text-success-foreground",
  },
  TERMINATED: {
    // TODO(tokens): no exact role for text-slate-700 (#475263)
    icon: <Icon icon={StopCircleIcon} size={16} className="text-slate-700" />,
    bgColor: "bg-slate-50",
    textColor: "!text-slate-700",
  },
  FAILED: {
    icon: (
      <Icon
        icon={CancelCircleIcon}
        size={16}
        className="text-error-foreground"
      />
    ),
    bgColor: "bg-error-surface",
    textColor: "!text-error-foreground",
  },
};

const completedWithErrorsStyle: StatusIconMap = {
  icon: (
    <Icon icon={Alert02Icon} size={16} className="text-warning-foreground" />
  ),
  bgColor: "bg-warning-surface",
  textColor: "!text-warning-foreground",
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
        "inline-flex items-center gap-1 rounded-md p-1",
        style.bgColor,
      )}
    >
      {style.icon}
      <Text
        variant="small-medium"
        className={cn(style.textColor, "capitalize")}
      >
        {label}
      </Text>
    </div>
  );
}
