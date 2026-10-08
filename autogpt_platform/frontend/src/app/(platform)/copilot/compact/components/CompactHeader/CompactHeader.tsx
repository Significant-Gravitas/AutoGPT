"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AgentStatusAvatar } from "@/components/molecules/AgentStatusAvatar/AgentStatusAvatar";
import type { AgentStatus } from "@/components/molecules/AgentStatusAvatar/helpers";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Menu01Icon, PencilEdit02Icon } from "@hugeicons/core-free-icons";

interface Props {
  agentStatus: AgentStatus;
  agentActivity: string;
  sessionId: string | null;
  sessions: SessionSummaryResponse[];
  onOpenSession: (id: string | null) => void;
  onOpenDetailedView: () => void;
}

export function CompactHeader({
  agentStatus,
  agentActivity,
  sessionId,
  sessions,
  onOpenSession,
  onOpenDetailedView,
}: Props) {
  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
      <AgentStatusAvatar
        status={agentStatus}
        name={AUTOPILOT_NAME}
        src={AUTOPILOT_AVATAR_URL}
        size="lg"
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <Text variant="body-medium" as="span">
          {AUTOPILOT_NAME}
        </Text>
        <Text
          variant="small"
          as="span"
          tone="muted"
          className="truncate"
          aria-live="polite"
          data-testid="agent-activity"
        >
          {agentActivity}
        </Text>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon-md" aria-label="Your chats">
            <Icon icon={Menu01Icon} size={18} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="max-h-96 w-72">
          <DropdownMenuLabel>Recent chats</DropdownMenuLabel>
          {sessions.length === 0 ? (
            <DropdownMenuLabel>No chats yet</DropdownMenuLabel>
          ) : null}
          {sessions.map((session) => (
            <DropdownMenuItem
              key={session.id}
              onSelect={() => onOpenSession(session.id)}
              className={session.id === sessionId ? "bg-muted" : undefined}
            >
              <span className="truncate">
                {session.title || "Untitled chat"}
              </span>
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={onOpenDetailedView}>
            Open detailed view
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Button
        variant="ghost"
        size="icon-md"
        aria-label="New chat"
        onClick={() => onOpenSession(null)}
      >
        <Icon icon={PencilEdit02Icon} size={18} />
      </Button>
    </header>
  );
}
