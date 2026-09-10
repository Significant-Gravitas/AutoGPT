import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatWeeklySpend } from "../../../helpers";
import { formatUntil } from "../../NowNext/helpers";
import { StatusBadge } from "./StatusBadge";

interface Props {
  agent: HomeAgentStatus;
}

const ACTION_CLASS = "h-7 min-w-0 rounded-md px-2.5 text-xs";

/** One expert with its status and two ways in. The detail line is gone:
 *  anything that needs doing is already spelled out under Needs you. */
export function AgentRow({ agent }: Props) {
  const spend = formatWeeklySpend(agent.spend_cents);
  const nextRun = agent.next_run_time
    ? `Next run ${formatUntil(agent.next_run_time)}`
    : null;
  const secondLine = [spend, nextRun].filter(Boolean).join(" · ");

  return (
    <div className="flex min-w-0 items-center gap-3 px-4 py-2.5">
      <ExpertAvatar
        name={agent.expert.name}
        avatarUrl={agent.expert.avatar_url}
        size={32}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text
            variant="body-medium"
            className="truncate leading-5 text-zinc-900"
          >
            {agent.expert.name}
          </Text>
          <StatusBadge status={agent.status} />
        </div>
        {secondLine ? (
          <Text
            variant="small"
            className="truncate tabular-nums leading-4 text-zinc-500"
            unmask={false}
          >
            {secondLine}
          </Text>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        <Button
          as="NextLink"
          href={`/copilot?expertId=${agent.expert.id}`}
          variant="secondary"
          size="small"
          className={ACTION_CLASS}
          aria-label={`Chat with ${agent.expert.name}`}
        >
          Chat
        </Button>
        <Button
          as="NextLink"
          href={`/team/${agent.expert.id}`}
          variant="secondary"
          size="small"
          className={ACTION_CLASS}
          aria-label={`Manage ${agent.expert.name}`}
        >
          Manage
        </Button>
      </div>
    </div>
  );
}
