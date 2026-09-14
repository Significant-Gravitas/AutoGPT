import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Button } from "@/components/atoms/Button/Button";
import { BubbleChatIcon, Settings01Icon } from "@hugeicons/core-free-icons";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatWeeklySpend } from "../../../helpers";
import { formatUntil } from "../../NowNext/helpers";
import { StatusBadge } from "./StatusBadge";

interface Props {
  agent: HomeAgentStatus;
}

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
            tone="primary"
            className="truncate leading-5"
          >
            {agent.expert.name}
          </Text>
          <StatusBadge status={agent.status} />
        </div>
        {secondLine ? (
          <Text
            variant="small"
            tone="muted"
            className="truncate tabular-nums leading-4"
            unmask={false}
          >
            {secondLine}
          </Text>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {/* Icon-only: the atom shows the aria-label as a hover tooltip. */}
        <Button
          as="NextLink"
          href={`/copilot?expertId=${agent.expert.id}`}
          variant="icon"
          size="icon-sm"
          leadingIcon={BubbleChatIcon}
          aria-label={`Chat with ${agent.expert.name}`}
        />
        <Button
          as="NextLink"
          href={`/team/${agent.expert.id}`}
          variant="icon"
          size="icon-sm"
          leadingIcon={Settings01Icon}
          aria-label={`Manage ${agent.expert.name}`}
        />
      </div>
    </div>
  );
}
