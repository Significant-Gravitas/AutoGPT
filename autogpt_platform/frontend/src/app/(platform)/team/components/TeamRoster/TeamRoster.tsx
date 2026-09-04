import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { Text } from "@/components/atoms/Text/Text";
import { ReactNode } from "react";
import { LIST_SURFACE_CLASS, SECTION_LABEL_CLASS } from "../../helpers";
import { AutopilotRow } from "../AutopilotRow";
import { ExpertRowSkeleton } from "../ExpertRowSkeleton";

interface Props {
  isLoading: boolean;
  podGroups: { pod: ExpertPod; experts: Expert[] }[];
  ungroupedExperts: Expert[];
  renderRow: (expert: Expert) => ReactNode;
}

interface SectionHeaderProps {
  name: string;
  count: number;
}

export function TeamRoster({
  isLoading,
  podGroups,
  ungroupedExperts,
  renderRow,
}: Props) {
  if (isLoading) {
    return (
      <div className={LIST_SURFACE_CLASS}>
        <AutopilotRow />
        {[0, 1, 2].map((index) => (
          <ExpertRowSkeleton key={index} />
        ))}
      </div>
    );
  }

  if (podGroups.length === 0) {
    return (
      <div className={LIST_SURFACE_CLASS}>
        <AutopilotRow />
        {ungroupedExperts.map(renderRow)}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <div className={LIST_SURFACE_CLASS}>
        <AutopilotRow />
      </div>
      {podGroups.map((group) => (
        <section key={group.pod.id} className="flex flex-col gap-2.5">
          <SectionHeader name={group.pod.name} count={group.experts.length} />
          {group.experts.length > 0 ? (
            <div className={LIST_SURFACE_CLASS}>
              {group.experts.map(renderRow)}
            </div>
          ) : (
            <Text variant="small" className="text-zinc-500">
              No experts in this pod yet.
            </Text>
          )}
        </section>
      ))}
      {ungroupedExperts.length > 0 ? (
        <section className="flex flex-col gap-2.5">
          <SectionHeader name="Ungrouped" count={ungroupedExperts.length} />
          <div className={LIST_SURFACE_CLASS}>
            {ungroupedExperts.map(renderRow)}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function SectionHeader({ name, count }: SectionHeaderProps) {
  return (
    <div className="flex items-baseline gap-2">
      <h3 className={SECTION_LABEL_CLASS}>{name}</h3>
      <Text variant="small" className="text-zinc-400">
        {count} {count === 1 ? "expert" : "experts"}
      </Text>
    </div>
  );
}
