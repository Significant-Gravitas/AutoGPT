import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatWeeklySpend } from "../../../helpers";
import { formatUntil } from "../../NowNext/helpers";
import { StatusBadge } from "./StatusBadge";

interface Props {
  agent: HomeAgentStatus;
}

export function AgentRow({ agent }: Props) {
  const spend = formatWeeklySpend(agent.spend_cents);
  return (
    <Link
      href={`/team/${agent.expert.id}`}
      className="group -mx-2 flex min-w-0 items-center gap-3 rounded-lg px-2 py-2.5 transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
    >
      <ExpertAvatar
        name={agent.expert.name}
        avatarUrl={agent.expert.avatar_url}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" className="truncate text-zinc-900">
            {agent.expert.name}
          </Text>
          <StatusBadge status={agent.status} />
        </div>
        <div className="flex min-w-0 items-center gap-1 text-zinc-500">
          <Text variant="small" className="min-w-0 truncate">
            {agent.detail}
          </Text>
          {spend ? (
            <Text
              variant="small"
              className="shrink-0 tabular-nums"
              unmask={false}
            >
              {`· ${spend}`}
            </Text>
          ) : null}
        </div>
      </div>
      {agent.next_run_time ? (
        <Text
          variant="small"
          className="shrink-0 tabular-nums text-zinc-400 group-hover:text-zinc-600"
        >
          {formatUntil(agent.next_run_time)}
        </Text>
      ) : (
        <Icon
          icon={ArrowRight01Icon}
          size={15}
          className="shrink-0 text-zinc-300 transition-transform group-hover:translate-x-0.5 group-hover:text-zinc-600"
          aria-hidden="true"
        />
      )}
    </Link>
  );
}
