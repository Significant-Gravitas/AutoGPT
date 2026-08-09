import Link from "next/link";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getExpertStatusLine } from "./helpers";
import { useTeamStrip } from "./useTeamStrip";

export function TeamStrip() {
  const { isVisible, hiredExperts, schedules } = useTeamStrip();

  if (!isVisible) return null;

  return (
    <div className="mb-6 flex snap-x gap-3 overflow-x-auto pb-2 text-left">
      {hiredExperts.map((expert) => (
        <TeamStripCard key={expert.id} expert={expert} schedules={schedules} />
      ))}
    </div>
  );
}

interface CardProps {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
}

function TeamStripCard({ expert, schedules }: CardProps) {
  return (
    <div className="flex w-56 shrink-0 snap-start flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-4">
      <Link
        href={`/team/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className="flex flex-col gap-2"
      >
        <ExpertAvatar name={expert.name} avatarUrl={expert.avatar_url} />
        <div className="min-w-0">
          <Text variant="body-medium" className="truncate">
            {expert.name}
          </Text>
          <Text variant="small" className="truncate text-zinc-500">
            {expert.role}
          </Text>
        </div>
        <Text variant="small" className="truncate text-zinc-500">
          {getExpertStatusLine(expert, schedules)}
        </Text>
      </Link>
      <Link
        href={`/copilot?expertId=${expert.id}`}
        aria-label={`Chat with ${expert.name}`}
        className="text-sm font-medium text-zinc-700 hover:text-zinc-900"
      >
        Chat
      </Link>
    </div>
  );
}
