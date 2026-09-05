import { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import Link from "next/link";
import { CardStat, CardStats } from "../../components/CardStats";

interface Props {
  experts: Expert[];
  scheduleCount: number;
  skillCount: number;
  workflowCount: number;
}

export function AutopilotSummaryCard({
  experts,
  scheduleCount,
  skillCount,
  workflowCount,
}: Props) {
  return (
    <aside
      aria-label="Team at a glance"
      className="flex flex-col gap-4 self-start rounded-xl border border-zinc-200 bg-white p-4"
    >
      <Text variant="large-medium" className="text-base text-zinc-700">
        Team at a glance
      </Text>
      <CardStats>
        <CardStat label="Experts">{experts.length}</CardStat>
        <CardStat label="Schedules">{scheduleCount}</CardStat>
        <CardStat label="Skills">{skillCount}</CardStat>
        <CardStat label="Workflows">{workflowCount}</CardStat>
      </CardStats>

      <section
        aria-label="Experts Autopilot works with"
        className="flex flex-col gap-2 border-t border-zinc-100 pt-4"
      >
        {experts.length === 0 ? (
          <Text variant="small" className="text-zinc-500">
            No experts hired yet. Autopilot works alone until you hire one.
          </Text>
        ) : (
          <ul className="flex flex-col gap-1">
            {experts.map((expert) => (
              <li key={expert.id}>
                <Link
                  href={`/team/${expert.id}`}
                  className="flex items-baseline justify-between gap-2 rounded-md px-2 py-1.5 hover:bg-zinc-50"
                >
                  <span className="truncate text-sm font-medium text-zinc-800">
                    {expert.name}
                  </span>
                  <span className="truncate text-xs text-zinc-500">
                    {expert.role}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
