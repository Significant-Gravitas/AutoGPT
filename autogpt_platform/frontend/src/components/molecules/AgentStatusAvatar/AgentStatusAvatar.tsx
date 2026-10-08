"use client";

import {
  Avatar,
  AvatarBadge,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import {
  AGENT_STATUS_BADGE_CLASS,
  AGENT_STATUS_LABEL,
  type AgentStatus,
  isAgentBusy,
} from "./helpers";

interface Props {
  status: AgentStatus;
  name: string;
  src?: string;
  size?: "sm" | "default" | "lg";
  className?: string;
}

export function AgentStatusAvatar({
  status,
  name,
  src,
  size = "default",
  className,
}: Props) {
  return (
    <Avatar
      size={size}
      role="img"
      aria-label={`${name}: ${AGENT_STATUS_LABEL[status]}`}
      data-status={status}
      className={cn(
        "transition-shadow duration-300",
        isAgentBusy(status) && "ring-2 ring-purple-200 ring-offset-1",
        status === "waiting" && "ring-2 ring-yellow-200 ring-offset-1",
        className,
      )}
    >
      {src ? <AvatarImage src={src} alt="" /> : null}
      <AvatarFallback>{name.charAt(0).toUpperCase()}</AvatarFallback>
      <AvatarBadge
        aria-hidden
        className={cn(
          "transition-colors duration-200",
          AGENT_STATUS_BADGE_CLASS[status],
        )}
      />
    </Avatar>
  );
}
