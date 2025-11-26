import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import {
  CheckCircleIcon,
  ClockIcon,
  PauseCircleIcon,
  StopCircleIcon,
  WarningCircleIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type StatusIconMap = {
  icon: React.ReactNode;
  bgColor: string;
  textColor: string;
};

const statusIconMap: Record<AgentExecutionStatus, StatusIconMap> = {
  INCOMPLETE: {
    icon: (
      <WarningCircleIcon size={16} className="text-red-700" weight="bold" />
    ),
    bgColor: "bg-red-50",
    textColor: "!text-red-700",
  },
  QUEUED: {
    icon: <ClockIcon size={16} className="text-yellow-700" weight="bold" />,
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  RUNNING: {
    icon: (
      <PauseCircleIcon size={16} className="text-yellow-700" weight="bold" />
    ),
    bgColor: "bg-yellow-50",
    textColor: "!text-yellow-700",
  },
  COMPLETED: {
    icon: (
      <CheckCircleIcon size={16} className="text-green-700" weight="bold" />
    ),
    bgColor: "bg-green-50",
    textColor: "!text-green-700",
  },
  TERMINATED: {
    icon: <StopCircleIcon size={16} className="text-slate-700" weight="bold" />,
    bgColor: "bg-slate-50",
    textColor: "!text-slate-700",
  },
  FAILED: {
    icon: <XCircleIcon size={16} className="text-red-700" weight="bold" />,
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
