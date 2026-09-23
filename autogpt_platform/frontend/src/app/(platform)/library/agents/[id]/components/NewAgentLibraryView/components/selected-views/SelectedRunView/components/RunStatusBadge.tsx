import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  Alert01Icon,
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
    icon: <Icon icon={AlertCircleIcon} size={16} className="text-red-700" />,
    bgColor: "bg-red-50",
    textColor: "!text-red-700",
  },
  QUEUED: {
    icon: <Icon icon={Clock01Icon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  RUNNING: {
    icon: <Icon icon={PauseCircleIcon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  REVIEW: {
    icon: <Icon icon={Alert01Icon} size={16} className="text-yellow-700" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  COMPLETED: {
    icon: (
      <Icon icon={CheckmarkCircle02Icon} size={16} className="text-green-700" />
    ),
    bgColor: "bg-green-50",
    textColor: "!text-green-700",
  },
  TERMINATED: {
    icon: <Icon icon={StopCircleIcon} size={16} className="text-slate-700" />,
    bgColor: "bg-slate-50",
    textColor: "!text-slate-700",
  },
  FAILED: {
    icon: <Icon icon={CancelCircleIcon} size={16} className="text-red-700" />,
    bgColor: "bg-red-50",
    textColor: "!text-red-700",
  },
};

type Props = {
  status: AgentExecutionStatus;
};

export function RunStatusBadge({ status }: Props) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-md p-1",
        statusIconMap[status].bgColor,
      )}
    >
      {statusIconMap[status].icon}
      <Text
        variant="small-medium"
        className={cn(statusIconMap[status].textColor, "capitalize")}
      >
        {status.toLowerCase()}
      </Text>
    </div>
  );
}
