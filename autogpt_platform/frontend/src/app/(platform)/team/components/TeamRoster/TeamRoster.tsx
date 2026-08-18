import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { Text } from "@/components/atoms/Text/Text";
import { ReactNode } from "react";
import { TEAM_GRID_CLASS } from "../../helpers";
import { AutopilotCard } from "../AutopilotCard";
import { ExpertTeamCardSkeleton } from "../ExpertTeamCardSkeleton";

interface Props {
  isLoading: boolean;
  podGroups: { pod: ExpertPod; experts: Expert[] }[];
  ungroupedExperts: Expert[];
  renderCard: (expert: Expert) => ReactNode;
}

interface SectionHeaderProps {
  name: string;
  count: number;
}

export function TeamRoster({
  isLoading,
  podGroups,
  ungroupedExperts,
  renderCard,
}: Props) {
  if (isLoading) {
    return (
      <div className={TEAM_GRID_CLASS}>
        <AutopilotCard />
        {[0, 1, 2].map((index) => (
          <ExpertTeamCardSkeleton key={index} />
        ))}
      </div>
    );
  }

  if (podGroups.length === 0) {
    return (
      <div className={TEAM_GRID_CLASS}>
        <AutopilotCard />
        {ungroupedExperts.map(renderCard)}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className={TEAM_GRID_CLASS}>
        <AutopilotCard />
      </div>
      {podGroups.map((group) => (
        <section key={group.pod.id} className="space-y-3">
          <SectionHeader name={group.pod.name} count={group.experts.length} />
          {group.experts.length > 0 ? (
            <div className={TEAM_GRID_CLASS}>
              {group.experts.map(renderCard)}
            </div>
          ) : (
            <Text variant="small" className="text-zinc-500">
              No experts in this pod yet.
            </Text>
          )}
        </section>
      ))}
      {ungroupedExperts.length > 0 ? (
        <section className="space-y-3">
          <SectionHeader name="Ungrouped" count={ungroupedExperts.length} />
          <div className={TEAM_GRID_CLASS}>
            {ungroupedExperts.map(renderCard)}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function SectionHeader({ name, count }: SectionHeaderProps) {
  return (
    <div className="flex items-baseline gap-2">
      <Text variant="h4">{name}</Text>
      <Text variant="small" className="text-zinc-500">
        {count} {count === 1 ? "expert" : "experts"}
      </Text>
    </div>
  );
}
